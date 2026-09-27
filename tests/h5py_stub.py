# Author: Paulo R. A. Gilz
"""A stand-in for h5py, holding only what the HDF5 reader actually touches.

h5py is not part of this project's base install -- adding a package to the
target machine costs a manual download of it and its whole transitive closure
-- and it is absent from both development environments. Without a stand-in the
HDF5 reader would be the one unreviewed path in the program, tested only on the
machine that can least easily run tests.

So this models the small surface `aoeviewer/hdf5.py` uses: a file that can be
written and reopened, groups that nest, 1-D datasets with a dtype, a shape and
attributes, and `visititems`. It keeps its trees in memory, keyed by path, and
writes the real HDF5 signature to disk so the reader's own sniffing works
against it.

It is not h5py. Where the two differ, the difference is noted in the code. It
is enough to prove the layout detection and the push into the core, and nothing
beyond that should be inferred from a passing test here.
"""

from __future__ import annotations

import numpy as np

# Trees by path, so a file written in a test can be opened again by the code
# under test -- which opens by path, as it would a real file.
_FILES = {}

SIGNATURE = b"\x89HDF\r\n\x1a\n"


class Dataset(object):
    def __init__(self, data, chunks=None, compression=None, attrs=None):
        self._values = np.asarray(data)
        self.chunks = chunks
        self.compression = compression
        self.compression_opts = None
        self.shuffle = False
        self.fletcher32 = False
        self.attrs = dict(attrs or {})

    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    @property
    def dtype(self):
        return self._values.dtype

    def __getitem__(self, key):
        return self._values[key]


class Group(object):
    def __init__(self, attrs=None):
        self.attrs = dict(attrs or {})
        self._children = {}

    # Real h5py raises on an existing name; this returns what is already there,
    # which only makes a test easier to write and cannot hide a bug in the
    # reader, since the reader never creates anything.
    def create_group(self, path):
        node = self
        for part in str(path).strip("/").split("/"):
            if part not in node._children:
                node._children[part] = Group()
            node = node._children[part]
        return node

    def create_dataset(self, name, data=None, **kwargs):
        parent = self
        parts = str(name).strip("/").split("/")
        for part in parts[:-1]:
            parent = parent.create_group(part)
        dataset = Dataset(data, chunks=kwargs.get("chunks"),
                          compression=kwargs.get("compression"))
        parent._children[parts[-1]] = dataset
        return dataset

    def __getitem__(self, path):
        node = self
        for part in str(path).strip("/").split("/"):
            node = node._children[part]
        return node

    def __contains__(self, path):
        try:
            self[path]
        except KeyError:
            return False
        return True

    def items(self):
        return list(self._children.items())


class File(Group):
    def __init__(self, path, mode="r"):
        Group.__init__(self)
        self._path = str(path)
        if mode == "r":
            existing = _FILES.get(self._path)
            if existing is None:
                raise OSError("no such file: %s" % self._path)
            self._children = existing._children
        else:
            _FILES[self._path] = self
            # The reader sniffs the signature before it opens anything, so the
            # bytes on disk have to be right even though the tree is not there.
            try:
                with open(self._path, "wb") as handle:
                    handle.write(SIGNATURE + b"\0" * 120)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def visititems(self, callback):
        def walk(prefix, group):
            for name, obj in sorted(group._children.items()):
                path = prefix + name
                if callback(path, obj) is not None:
                    return True
                if isinstance(obj, Group) and walk(path + "/", obj):
                    return True
            return False

        walk("", self)
