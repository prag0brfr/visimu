#!/usr/bin/env python3
# Author: Paulo R. A. Gilz
"""Generate an AOE-format sample file, for trying the app out and benchmarking.

    python tools/gen_sample.py sample.aoe --sims 8 --rows 50000
"""

from __future__ import annotations

import argparse
import math
import os


# The units the indexed dialect declares, in CHANNEL order. Empty means the
# quantity is dimensionless, and the file writes '' for it -- which is the case
# a parser is most likely to get wrong.
CHANNELS = ("TIME", "ALPHA", "MACH", "NZ", "ALT", "THETA")
UNITS = ("s", "deg", "", "", "ft", "deg")


def generate(path: str, sims: int, rows: int, wrap: bool = False,
             indexed: bool = False, entity: bool = False) -> None:
    """Write a sample file.

    `indexed` selects the other *PARAMETRES dialect -- one parameter per line,
    prefixed by its ordinal and followed by its unit:

        \t1\t'TIME'\t\t's'
        \t2\t'ALPHA'\t \t'deg'

    `entity` adds an *ENTITY section of free text between the parameters and
    the first table, which real files use for provenance notes and which must
    be skipped rather than parsed.
    """
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("*PARAMETRES\t\t\t\n" if indexed else "*PARAMETRES\n")
        if indexed:
            for index, (name, unit) in enumerate(zip(CHANNELS, UNITS), start=1):
                handle.write("\t{0}\t'{1}'\t \t'{2}'\n".format(index, name, unit))
        else:
            handle.write(
                " 'TIME'    'ALPHA'   'MACH'    'NZ'      'ALT'     'THETA'\n"
            )
        handle.write("\n")

        if entity:
            handle.write("*ENTITY\t\t\t\n")
            handle.write("\n")
            handle.write("Free text, any amount of it, 123 and symbols !@#\n")
            handle.write("\n")

        for s in range(sims):
            if indexed:
                handle.write("*TABLEAU NUM SIMU: {0}.2.0\n".format(s + 1))
            else:
                handle.write("*TABLEAU : RUN_{0:03d}\n".format(s + 1))
            mach = 0.60 + 0.03 * s
            alt = 5000.0 + 1000.0 * s
            for r in range(rows):
                t = r * 0.01
                alpha = 2.0 + math.sin(t * 0.7) * (1.0 + 0.1 * s)
                nz = 1.0 + 0.3 * math.cos(t * 0.4)
                theta = 5.0 * math.sin(t * 0.2) + 0.5 * s
                values = (t, alpha, mach, nz, alt, theta)
                if wrap:
                    handle.write(" {0:.6f} {1:.6f} {2:.6f}\n".format(*values[:3]))
                    handle.write(" {0:.6f} {1:.6f} {2:.6f}\n".format(*values[3:]))
                else:
                    handle.write(
                        " {0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f}\n".format(*values)
                    )
        handle.write("*FIN\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--sims", type=int, default=6)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="split each record over two lines, as Fortran does when the "
             "format width overflows",
    )
    parser.add_argument(
        "--indexed",
        action="store_true",
        help="use the indexed *PARAMETRES dialect: one parameter per line, "
             "with its ordinal and its unit",
    )
    parser.add_argument(
        "--entity",
        action="store_true",
        help="add an *ENTITY block of free text before the first table",
    )
    args = parser.parse_args()

    generate(args.output, args.sims, args.rows, args.wrap,
             args.indexed, args.entity)
    size = os.path.getsize(args.output)
    print(
        "wrote {0}  ({1} runs x {2} rows, {3:.1f} MB)".format(
            args.output, args.sims, args.rows, size / (1024 * 1024)
        )
    )


if __name__ == "__main__":
    main()
