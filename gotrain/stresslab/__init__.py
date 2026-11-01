"""StressLab: robustness evaluation for an already-trained checkpoint.

See gotrain.stresslab.perturbations (pure numpy, fully testable without
torch) for the perturbation battery, and gotrain.stresslab.runner (needs
torch + geobench + the real downloaded dataset) for the model-assisted
evaluation that actually scores a checkpoint under each one.
"""
