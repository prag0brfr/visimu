# aoeViewer 2.1.0 — user-guide addendum

This note updates the 2.0.0 User and Reference Guide shipped as
`aoeViewer-2.0.0-User-and-Reference-Guide.pdf`. The application version is
**2.1.0**. Compile commands in `build.sh` / `build.bat` are unchanged.

## Dual encoding (STYLE)

A curve now has two independent identities:

- **Colour by** — Simulation, Parameter, or ISO. Sets the colour only.
- **Style by** — None, File, Simulation, or ISO. Sets the dash pattern and
  marker. *None* keeps dash on the colour key (2.0.0 behaviour).

Typical pairing for several files on one tile: *Colour by Parameter* +
*Style by File*. *Apply encoding to all tiles* copies both choices.

The STYLE tab lists those identities in two tables, **Colour** and **Line**.
Each table scrolls when the list is long.

## Plot titles and axes

Axis captions next to Y and under X are blank. The tile title names every
selected parameter as `ALPHA, NZ = f(TIME)`. Long lists wrap onto two lines
instead of being truncated.

## PARAMS order

On the active tile, selected Y parameters appear first (in the order they
were ticked). The filter still only hides names that do not match; it does
not change that order. A selected name that fails the filter stays selected
and stays at the top, hidden until the filter lets it through.

## Workspaces

The JSON schema is still version 3. New files also store `app_version`
(`2.1.0`) and per-tile `style_mode` / `style_iso_param`. A workspace written
by 2.0.0 opens with *Style by* = None. Markers, areas, text notes, ISO
filters, 3D views, inspect, scales and aliases are unchanged.
