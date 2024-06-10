# Makefile for running audio_metadata.py, dataset.py, then sound_classifier.py

# Define the Python interpreter
PYTHON = python3

# Define the scripts
METADATA_SCRIPT = audio_metadata.py
DATASET_SCRIPT = dataset.py
CLASSIFIER_SCRIPT = sound_classifier.py

# Default target
all: run_metadata run_dataset run_classifier

# Target to run audio_metadata.py
run_metadata:
	$(PYTHON) $(METADATA_SCRIPT)

# Target to run dataset.py
run_dataset: run_metadata
	$(PYTHON) $(DATASET_SCRIPT)

# Target to run sound_classifier.py
run_classifier: run_dataset
	$(PYTHON) $(CLASSIFIER_SCRIPT)

# Clean target (if you need to clean up any generated files, define it here)
clean:
	rm -f train_dl.pth val_dl.pth
	rm -rf __pycache__

# Phony targets
.PHONY: all run_metadata run_dataset run_classifier clean
