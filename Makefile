install:
	pip3 install -e .

run-tests:
	python3 -m pytest

run-benchmark:
	python3 ./benchmark/tst.py