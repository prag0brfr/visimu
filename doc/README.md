# doc/

`aoeViewer-2.2.3-User-and-Reference-Guide.tex` is the manual; the PDF beside it
is what that source builds to. It replaces both the 2.0.0 guide and the 2.1.x
addenda — everything those said is in here, with the passages later releases
invalidated rewritten rather than annotated.

To rebuild after editing the source:

    pdflatex aoeViewer-2.2.3-User-and-Reference-Guide.tex

three times, so the table of contents and the cross-references settle. It needs
nothing beyond a standard TeX Live: report, fontenc, inputenc, lmodern,
mathpazo, microtype, geometry, booktabs, longtable, array, xcolor, enumitem,
fancyhdr, titlesec, tcolorbox, amsmath, fancyvrb, hyphenat and hyperref.

Conventions inside the guide: a blue box marks behaviour introduced or changed
after 2.0.0 and names the version, a grey box records something the source says
that the prose might not, and a red box records a real limitation.
