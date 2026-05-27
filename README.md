# Online Amplitude-Encoding Quantum Reservoir Computing

This repository contains companion code for reproducing the numerical and hardware results of the paper *Online processing with an amplitude encoding scheme*.

The code is organized to separate reusable utilities, data-generation tutorials, and figure reproduction from precomputed data.


## Files

- `utilities.py`: basic Hilbert-space, composite-space, and density-matrix utilities.
- `online_amplitude_encoding_tutorial.ipynb`: tutorial notebook showing how to construct the online amplitude-encoding protocol and generate data.
- `reproduce_figures.ipynb`: notebook that loads precomputed data and reproduces the paper figures.
- `online_FN_qrc_env.yml`: conda environment file listing the required Python packages and versions.
- `sk_Santa_Fe_2000.npy`: Santa Fe dataset.

## Data

The data required by `reproduce_figures.ipynb` should be downloaded from the associated Zenodo record, https://doi.org/10.5281/zenodo.20399163.

## Recommended workflow

1. Install the required Python packages in your environment or directly install the provided environment file.
2. Download the Zenodo data and set `DATA_DIR` and `DATASET_DIR`.
3. Run `reproduce_figures.ipynb` to regenerate the paper figures.
4. Use `online_amplitude_encoding_tutorial.ipynb` to inspect or reproduce the data-generation workflow.

## Notes

The tutorial includes two model choices:

- an all-to-all spin-network model, used for the Fig. 2-style benchmark;
- a nearest-neighbor hardware-tailored model, used for simulator and hardware execution.
