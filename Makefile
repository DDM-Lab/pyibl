.PHONY: dist clean upload test doc

dist:	clean test doc
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist/*

doc/_downloads/binary-choice.zip:
	cd examples/binary-choice/ ; zip ../../doc/_downloads/binary-choice.zip *.py requirements.txt

doc:
	cd doc/ ; make html

upload: dist
	twine upload dist/*
	scp -r doc/_build/html/* dfm@janus.hss.cmu.edu:/var/www/pyibl/

test:
	pytest
	xdg-open compare-plots.html
