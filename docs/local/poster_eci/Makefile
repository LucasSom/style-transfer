BUILD := \
	poster \


DEPS := \
	beamerthemegemini.sty \
	beamercolorthememitmarch.sty \


LATEX  := lualatex
BIBTEX := bibtex

LATEXOPTS := -interaction nonstopmode

TARGETS := $(patsubst %, %.pdf, $(BUILD)) Miguel_Martin_PosterNMVII.pdf Miguel_Martin_PosterNMVII.jpg Miguel_Martin_PosterNMVII_thumbnail.jpg

# phony targets

all: $(TARGETS)

clean:
	rm -rf *.pdf *.aux *.bbl *.blg *.log *.nav *.out *.snm *.toc

.PHONY: all clean

# main targets

poster.pdf: poster.tex poster.bib $(DEPS)
	$(eval SRC_$@ = $(patsubst %.tex, %, $<))
	$(LATEX) $(LATEXOPTS) $(SRC_$@)
	$(BIBTEX) $(SRC_$@)
	$(LATEX) $(LATEXOPTS) $(SRC_$@)
	$(LATEX) $(LATEXOPTS) $(SRC_$@)

poster.bib: poster.aux
	bibtex $<

Miguel_Martin_PosterNMVII.pdf: poster.pdf
	cp $< $@

Miguel_Martin_PosterNMVII.jpg: poster.pdf
	pdftoppm -jpeg $< poster 
	mv poster-1.jpg $@

Miguel_Martin_PosterNMVII_thumbnail.jpg: Miguel_Martin_PosterNMVII.jpg
	convert $< -resize 25% $@
