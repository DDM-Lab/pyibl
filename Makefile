.PHONY: dist clean upload test doc

dist:	clean test doc
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist/*

doc/_downloads/binary-choice.zip:
	cd examples/binary-choice/ ; zip ../../doc/_downloads/binary-choice.zip *.py requirements.txt

doc/_downloads/rps.zip:
	cd examples/rps/ ; zip ../../doc/_downloads/rps.zip *.py requirements.txt

doc: doc/_downloads/binary-choice.zip doc/_downloads/rps.zip
	cd doc/ ; make html

publish: doc
	scp -r doc/_build/html/* dfm@janus.hss.cmu.edu:/var/www/pyibl/

upload: dist publish
	twine upload dist/*

test:
	pytest
	xdg-open compare-plots.html
