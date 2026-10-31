# doc/

`visimu-2.4.0-User-and-Reference-Guide.tex` is the manual; the PDF beside it is
what that source builds to. It describes the current version and nothing else:
where behaviour has changed, the text was rewritten rather than appended to.

To rebuild after editing the source:

    pdflatex visimu-2.4.0-User-and-Reference-Guide.tex

three times, so the table of contents and the cross-references settle. It needs
nothing beyond a standard TeX Live: report, fontenc, inputenc, lmodern,
mathpazo, microtype, geometry, booktabs, longtable, array, xcolor, enumitem,
fancyhdr, titlesec, tcolorbox, amsmath, fancyvrb, hyphenat and hyperref.

Conventions inside the guide: a grey box records something worth knowing to use
the program well -- usually why a thing behaves as it does -- and a red box
records a real limitation. There are no figures; it is a text reference.
