.PHONY: dist clean

dist:	clean
	python -m build -n

clean:
	rm -rf dist/*
