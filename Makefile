#  Usefull command
PROJECT_NAME = project

setup-env:
	conda create -n ${PROJECT_NAME} python=3.8 -y

install-all:
	pip install -r requirements-dev.txt -r requirements.txt

run-test:
	python -m pytest test/

format-all:
	black src/
	black test/
