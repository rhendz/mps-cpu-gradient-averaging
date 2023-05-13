# Makefile for creating a Python virtual environment and installing requirements

# Name of the virtual environment
VENV = .venv

# Command to create a virtual environment
VENV_CMD = python3 -m venv $(VENV)

# Name of the requirements file
REQUIREMENTS = requirements.txt

# Command to install requirements
INSTALL_REQS_CMD = $(VENV)/bin/pip install -r $(REQUIREMENTS)

.PHONY: venv install

# Target to create a virtual environment
venv:
	$(VENV_CMD)

# Target to install requirements
install:
	$(INSTALL_REQS_CMD)
