# Experiment setup

The experiment modules use the AdversariaLLM toolkit and the model wrappers
in adapter/. Install the toolkit's dependencies in a virtual environment.

Benchmarks need the behavior datasets and optimizer targets in AdversariaLLM/data/.
GPU experiments require CUDA; checkpoint access may require Hugging Face login.

Repository paths and run settings are configured by the experiment modules.
