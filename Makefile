.PHONY: dist clean upload test

dist:	clean test
	python -m build -n
	cd doc/ ; make html

clean:
	rm -rf dist/*

upload: dist
	twine upload dist/*
	scp -r doc/_build/html/* dfm@janus.hss.cmu.edu:/var/www/pyibl/

test:
	pytest
	xdg-open compare-plots.html
