boring-template

## Install

```bash
# Change in Makefile the project name
make setup-env
conda activate <project_name>
make install-all
```


## Test

```bash
make run-test
```

## Format

```bash
make format-all
```

## Architecture

```
├── CONTRIBUTING.md <- Coding rules for devs
├── data <- Data folder
├── doc <- Documentation
├── Dockerfile
├── Makefile <- Usefull command
├── README.md
├── requirements-dev.txt <- Linting etc ...
├── requirements.txt <- Package dependencies
├── result
├── setup.cfg <- Linting rules etc ...
├── src
│   └── sandbox <- Prototyping folder
└── test
```