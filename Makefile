#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y glucoza || :
	@pip install --upgrade pip cython wheel setuptools
	@pip install -r requirements.txt
	@pip install -e .
	@pip freeze | grep glucoza
