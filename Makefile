init:
	pip install -r requirements.txt

test:
	python setup.py test

sdist:
	python setup.py sdist --formats=gztar,zip

register:
	python setup.py register

upload:
	python setup.py sdist --formats=gztar,zip upload

clean:
	rm -f *~
	find . -name "*.pyc" -delete
	rm -rf ./*.egg-info
	rm -rf ./dist