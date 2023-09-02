.PHONY: dist clean upload

dist:	clean
	python -m build -n
	cd doc/ ; make html

clean:
	rm -rf dist/*

upload: dist
	twine upload -u dfmorrison dist/*
	scp -r doc/_build/html/* dfm@janus.hss.cmu.edu:/var/www/pyibl/
