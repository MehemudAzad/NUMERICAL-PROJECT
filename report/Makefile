# Build the CSE-402 report.
#   make          -> main.pdf (latexmk resolves refs/TikZ externalisation)
#   make clean    -> remove build artifacts, keep the PDF
#   make cleanall -> also remove the PDF
#   make tectonic -> main.pdf with Tectonic (no TeX install needed; `brew install tectonic`)
#
# Figures are read from ../figures/. Regenerate them with ../run_all.sh
# (Tier-3 figures come from the Kaggle notebook and are committed).

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(wildcard ../figures/*.png)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

cleanall:
	latexmk -C $(MAIN).tex

tectonic:
	tectonic -X compile $(MAIN).tex

.PHONY: all clean cleanall tectonic
