# Author: Paulo R. A. Gilz
"""A stand-in for h5py, holding only what the HDF5 reader actually touches.

h5py is not part of this project's base install -- adding a package to the
target machine costs a manual download of it and its whole transitive closure
-- and it is absent from both development environments. Without a stand-in the
HDF5 reader would be the one unreviewed path in the program, tested only on the
machine that can least easily run tests.

So this models the small surface `visimu/hdf5.py` uses: a file that can be
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
        self._tree_path = ""

    # Real h5py raises on an existing name; this returns what is already there,
    # which only makes a test easier to write and cannot hide a bug in the
    # reader, since the reader never creates anything.
    def create_group(self, path):
        node = self
        for part in str(path).strip("/").split("/"):
            if part not in node._children:
                child = Group()
                child._tree_path = "%s/%s" % (
                    getattr(node, "_tree_path", ""), part)
                node._children[part] = child
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
        COUNTS["getitem"] += 1
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

    # Real h5py groups are mappings and know their own absolute path. Both are
    # read by the reader -- keys() to list a file's channels and runs without
    # walking anything, and name to build a dataset's path from its channel
    # and its run -- so the stand-in has to have them or it tests a route the
    # application never takes.
    def keys(self):
        return list(self._children)

    def __iter__(self):
        return iter(self._children)

    @property
    def name(self):
        return "/" + getattr(self, "_tree_path", "").strip("/")


# What the reader did to the file, so a test can measure the cost of an open
# rather than describe it. The whole point of working the layout out from
# names is that it opens a handful of datasets instead of all of them, and
# that is a number.
COUNTS = {"file_open": 0, "getitem": 0, "visititems": 0, "links_visit": 0}


def reset_counts():
    for key in COUNTS:
        COUNTS[key] = 0


class _Links(object):
    """h5py's h5l.LinkProxy, as far as the reader uses it.

    visit() reports every path in the file, groups included, and never opens
    the object a link points to -- which is the whole reason the reader
    prefers it to visititems().
    """

    def __init__(self, owner):
        self._owner = owner
        self.opened = 0          # stays zero: visiting opens nothing

    def visit(self, callback):
        COUNTS["links_visit"] += 1

        def walk(prefix, group):
            for name, obj in sorted(group._children.items()):
                path = prefix + name
                if callback(path) is not None:
                    return True
                if isinstance(obj, Group) and walk(path + "/", obj):
                    return True
            return False

        walk("", self._owner)


class _FileId(object):
    def __init__(self, owner):
        self._owner = owner

    @property
    def valid(self):
        return not getattr(self._owner, "_closed", False)

    @property
    def links(self):
        return _Links(self._owner)


class File(Group):
    def __init__(self, path, mode="r"):
        Group.__init__(self)
        self._path = str(path)
        if mode == "r":
            COUNTS["file_open"] += 1
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
        self.close()
        return False

    def close(self):
        self._closed = True

    @property
    def id(self):
        """The low-level handle, with the two pieces the reader uses.

        `valid` is how the handle cache tells a live handle from one that has
        been closed, and `links.visit` is the cheap traversal: it walks the
        link names without opening the objects they point to, which is what
        makes working the layout out from paths cost nothing per dataset.
        """
        return _FileId(self)

    def visititems(self, callback):
        COUNTS["visititems"] += 1

        def walk(prefix, group):
            for name, obj in sorted(group._children.items()):
                path = prefix + name
                if callback(path, obj) is not None:
                    return True
                if isinstance(obj, Group) and walk(path + "/", obj):
                    return True
            return False

        walk("", self)
