# rgc-ei-fit-hh

Code accompanying ICML 2026: **Learning Biophysical Models of Large-Scale Multineuronal Data to Enable Precise Neurostimulation**

Amrith Lotlikar, Ian Christopher Tanoh, Praful Vasireddy, Andrew Lanpouthakoun, Ramandeep Vilkhu, Michael Sommeling, A.J. Phillips, Alexander Sher, Alan Litke, Scott W. Linderman, E.J. Chichilnisky, Subhasish Mitra

This repository provides **`rgc_models`** and **`splinex`**, two installable Python packages for building biophysical digital twins of retinal ganglion cells from extracellular multi-electrode array recordings. The framework infers multi-compartment Hodgkin–Huxley parameters from electrical-image features and stimulation thresholds using differentiable simulation in [JAXLEY](https://github.com/jaxleyverse/jaxley).

## Installation

Requires Python 3.10+ (tested with Python 3.10.18). Clone this repository, `cd` into it, and use a virtual environment. For example:

```bash
cd rgc-ei-fit-hh          # repository root (contains setup.py)
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install .               # `.` = this directory
```

`pip install .` installs from the repository root (the `.` is a relative path to the current directory). It registers **`rgc_models`** and **`splinex`** and installs pinned dependencies from `requirements.txt` (notably **jaxley==0.9.0**, jax/jaxlib 0.6.1). For GPU/TPU, install matching `jax`/`jaxlib` wheels first if needed: [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

```bash
python -c "from rgc_models.rgc_model import RGC; from rgc_models.elec_coords import ELECTRODE_LOCATIONS_UM; print(ELECTRODE_LOCATIONS_UM.shape)"
```

## Packages

### `rgc_models`

| File | Description |
|------|-------------|
| `rgc_model.py` | Main `RGC` class: heterogeneous multi-compartment HH morphology (soma, axon hillock, sodium channel band, narrow region, distal axon), forward EI simulation, electrical-stimulation inference, and feature-based loss computation. Axon geometry is parameterized by 3D B-spline control points. |
| `lib.py` | Ion channels (HH sodium/potassium/leak, calcium dynamics), triphasic stimulus waveforms, point-source extracellular potentials, and auxiliary simulator utilities. |
| `loss_functions.py` | Differentiable EI feature extractors and geometry regularizers: sodium/potassium/capacitive peaks, propagation-delay features, EI widths, and spline/control-point smoothness penalties. |
| `rgc_location_initialization.py` | Data-driven initialization of soma position and axon trajectory from recorded EIs (axon-vs-soma classification, waypoint extraction, B-spline resampling). |
| `plotting.py` | Visualization helpers for multi-electrode stimulation response maps. |
| `elec_coords.py` | Loader for the paper's 512-channel, 30 µm-pitch MEA layout (`mea_512ch_30um_pitch_elec_coords.npy`, um, xyz). Import via `from rgc_models.elec_coords import ELECTRODE_LOCATIONS_UM`. |

### `splinex`

Vendored subset of [**splinex**](https://github.com/cornelius-braun/splinex) (Braun et al.): differentiable B-spline interpolation in JAX. Axon compartment locations are obtained by cubic B-spline interpolation through a low-dimensional set of optimized control points (first point fixed at the soma), rather than fitting each compartment position independently. Included here so the release is self-contained.

## Citation

If you use this code, please cite our ICML 2026 paper (bibtex forthcoming).

## License

MIT License — see [LICENSE](LICENSE).
