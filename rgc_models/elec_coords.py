"""Electrode coordinates for the 512-channel, 30 µm-pitch MEA (um, xyz)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Paper: Figure Electrical_Image — "512-channel, 30 µm-pitch MEA".
# Extracted from `data/final_triplet_dataset.h5`, cell_0. Layout is identical across cells.
_DATA_FILE = "mea_512ch_30um_pitch_elec_coords.npy"
_ELECTRODE_LOCATIONS_UM = np.load(Path(__file__).with_name(_DATA_FILE))
ELECTRODE_LOCATIONS_UM = jnp.asarray(_ELECTRODE_LOCATIONS_UM)
