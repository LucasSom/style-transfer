plan.pdf: main.tex
	pdflatex $<
	bibtex main.aux
	pdflatex $<
	pdflatex $<
	mv main.pdf $@
