run-tests:
	PYTHONPATH=./src pytest
	pytest ./tests


install:
	pip3 install -e .

run-tests:
	python3 -m pytest