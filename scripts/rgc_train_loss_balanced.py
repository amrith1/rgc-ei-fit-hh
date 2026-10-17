from rgc_models.rgc_model import RGC
from rgc_models.rgc_location_initialization import plot_ei_facts, initial_guess_axon_location, get_ais_electrode
from rgc_models.plotting import plot_stimulation_cross_sections
import matplotlib.pyplot as plt
import numpy as np
import optax
import jax
import jax.numpy as jnp
import time
from tqdm import tqdm
import os
import pickle

import argparse
import h5py

jax.config.update('jax_platform_name', 'cpu')  # Force CPU usage
jax.config.update('jax_enable_x64', True)

ALL_ABLATABLE_LOSS_FEATURES = frozenset({
    'sodium_peaks',
    'diffusion_peaks',
    'potassium_peaks',
    'sodium_velocity',
    'ei_widths_so_po',
    'estim',
})


def parse_loss_features(spec: str | None) -> frozenset[str]:
    """Parse --loss_features comma list; default all six ablatable terms."""
    if spec is None or spec.strip() == '':
        return ALL_ABLATABLE_LOSS_FEATURES
    enabled = frozenset(part.strip() for part in spec.split(',') if part.strip())
    unknown = enabled - ALL_ABLATABLE_LOSS_FEATURES
    if unknown:
        raise ValueError(
            f"Unknown --loss_features entries: {sorted(unknown)}. "
            f"Allowed: {sorted(ALL_ABLATABLE_LOSS_FEATURES)}"
        )
    if not enabled:
        raise ValueError("--loss_features must include at least one feature (use initial-guess script for no training)")
    return enabled


def apply_loss_feature_ablation(cell_obj, enabled_loss_features: frozenset[str]) -> None:
    """Zero ablated weights and skip their forward/backward in loss_v1."""
    for feature_name in list(cell_obj.ei_feature_weights.keys()):
        if feature_name not in enabled_loss_features:
            cell_obj.ei_feature_weights[feature_name] = 0.0
    if 'estim' not in enabled_loss_features:
        cell_obj.estim_loss_weight = 0.0
    cell_obj.set_training_loss_features(enabled_loss_features)


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _save_for_evaluation_pickle(output_dir: str, tag: str, cell_id: int, piece: str, RGC_kwargs: dict, full_params: dict) -> str:
    """
    Save a minimal pickle for downstream evaluation scripts.
    Format matches other *_trained.pkl artifacts: {'RGC_kwargs': ..., 'full_params': ...}.
    """
    for_eval_dir = os.path.join(output_dir, "for_evaluation")
    _mkdir(for_eval_dir)
    out_path = os.path.join(for_eval_dir, f"{tag}_{cell_id}_{piece}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"RGC_kwargs": RGC_kwargs, "full_params": full_params}, f)
    return out_path

PIECE = '2020-10-06-7'
CELL_ID = 244
# Output directory structure
OUTPUT_DIR = 'triplet_results_no_name'

def get_cell_index(dataset_path, piece, cell_id):
    with h5py.File(dataset_path, 'r') as h5f:
        for cell_idx in range(1000):
            piece = h5f['cells'][f'cell_{cell_idx}'].attrs['piece']
            cell_id = h5f['cells'][f'cell_{cell_idx}'].attrs['cell_id_piece']
            if piece == PIECE and cell_id == CELL_ID:
                return cell_idx
            if cell_idx == 1000:
                raise ValueError(f"Cell {cell_id} {piece} not found in dataset")

MODE = 'triplet_og' 
TRAIN_FRACTION = 1.0
SKIP_BIELEC = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--piece', type=str, required=True)
    parser.add_argument('--cell_id', type=int, required=True)
    parser.add_argument('--mode', type=str)
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="Path to the HDF5 dataset. Must contain 'triplet' for --mode triplet_og and 'single' for --mode single_elec.",
    )
    parser.add_argument('--train_fraction', type=float, required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--skip_bielec', action='store_true', help='Skip bielec inference and plotting')
    parser.add_argument('--max_num_thresholds', type=int, default=10, help='Maximum number of thresholds to use for training (default: 10)')
    parser.add_argument(
        '--loss_features',
        type=str,
        default=None,
        help=(
            "Comma-separated enabled loss terms (ablated terms zeroed after balancing). "
            "Choices: sodium_peaks, diffusion_peaks, potassium_peaks, sodium_velocity, "
            "ei_widths_so_po, estim. Default: all six."
        ),
    )
    args = parser.parse_args()

    if args.piece is not None and args.cell_id is not None:
        PIECE, CELL_ID = args.piece, args.cell_id
    else:
        raise ValueError("Must provide both piece and cell_id")

    MODE = args.mode
    OUTPUT_DIR = args.output_dir
    DATASET_PATH = args.dataset_path
    TRAIN_FRACTION = args.train_fraction
    SKIP_BIELEC = args.skip_bielec
    MAX_NUM_THRESHOLDS = args.max_num_thresholds
    ENABLED_LOSS_FEATURES = parse_loss_features(args.loss_features)


assert MODE in ['triplet_og', 'single_elec']
#if mode is single_elec, assrt train_fraction is between 0 and 1
if MODE == 'single_elec':
    assert TRAIN_FRACTION is not None and TRAIN_FRACTION >= 0.0 and TRAIN_FRACTION <= 1.0

dataset_path_lower = str(DATASET_PATH).lower()
if MODE == 'single_elec':
    assert 'single' in dataset_path_lower, (
        f"For --mode single_elec, dataset_path must contain 'single'. Got: {DATASET_PATH}"
    )
elif MODE == 'triplet_og':
    assert 'triplet' in dataset_path_lower, (
        f"For --mode triplet_og, dataset_path must contain 'triplet'. Got: {DATASET_PATH}"
    )

CELL_INDEX = get_cell_index(DATASET_PATH, PIECE, CELL_ID)
print(f"Cell index: {CELL_INDEX}")

cell_id_piece = f"{CELL_ID}-{PIECE}"
cell_folder = os.path.join(OUTPUT_DIR, 'cells_data', f'cell_{cell_id_piece}')
_mkdir(cell_folder)
_mkdir(os.path.join(cell_folder, 'plots'))
_mkdir(os.path.join(cell_folder, 'data'))

print(f"Training cell {cell_id_piece}, cell index {CELL_INDEX}")

CONTROL_POINT_SPACING = 40.0
AXON_COMP_LENGTH = 4.0
ARRAY_OVERSHOOT_LENGTH = 500.0
TIMING_FEATURES_MIN_AMP_mV = 0.01 #roughly 3 ADC counts

with h5py.File(DATASET_PATH, 'r') as h5f:
    cells_group = h5f['cells']
    cell_ei = cells_group[f'cell_{CELL_INDEX}']['ei_mV_raw_20kHz'][:]
    cell_ei_100kHz = cells_group[f'cell_{CELL_INDEX}']['ei_mV_100kHz'][:]
    coords = cells_group[f'cell_{CELL_INDEX}']['elec_coords'][:]
    #piece is an attribute
    piece = cells_group[f'cell_{CELL_INDEX}'].attrs['piece']
    cell_id = cells_group[f'cell_{CELL_INDEX}'].attrs['cell_id_piece']
    # `final_triplet_dataset.h5` intentionally omits cell_type; it is not used for training.
    cell_type = 'unknown'

if MODE == 'triplet_og':
    with h5py.File(DATASET_PATH, 'r') as h5f:
        cells_group = h5f['cells']
        cell_group = cells_group[f'cell_{CELL_INDEX}']
        triplet_stim_elecs = cell_group.attrs['stim_elecs'][:]
        amps_gsort = cell_group['amps_gsort'][:]
        probs_multisite = cell_group['probs_multisite'][:]
        # Final triplet dataset includes these directly (length-3, aligned with stim_elecs).
        # Fail loudly if missing in triplet mode.
        assert 'pos_thresholds_uA' in cell_group, "Missing dataset pos_thresholds_uA in triplet H5"
        assert 'neg_thresholds_uA' in cell_group, "Missing dataset neg_thresholds_uA in triplet H5"
        pos_thresholds_uA = cell_group['pos_thresholds_uA'][:]
        neg_thresholds_uA = cell_group['neg_thresholds_uA'][:]
    print(f"Triplet stimulation electrodes: {triplet_stim_elecs}")

elif MODE == 'single_elec':
    with h5py.File(DATASET_PATH, 'r') as h5f:
        cells_group = h5f['cells']
        train_test_ordering_full = cells_group[f'cell_{CELL_INDEX}']['train_test_ordering'][:]
        # Limit to first MAX_NUM_THRESHOLDS (or all if fewer are available)
        num_available = len(train_test_ordering_full)
        num_to_use = min(MAX_NUM_THRESHOLDS, num_available)
        train_test_ordering = train_test_ordering_full[:num_to_use]
        stim_thresholds = cells_group[f'cell_{CELL_INDEX}']['thresholds_gsort_raw'][:]
        if num_available > MAX_NUM_THRESHOLDS:
            print(f"Limited train_test_ordering from {num_available} to {num_to_use} (max_num_thresholds={MAX_NUM_THRESHOLDS})")
        else:
            print(f"Using all {num_available} available thresholds (max_num_thresholds={MAX_NUM_THRESHOLDS})")

axes = plot_ei_facts(cell_ei, coords)
axes[0].set_title(f'Cell {CELL_INDEX} local ID {piece} {cell_id} {cell_type}')
axon_control_points, axon_control_points_raw, final_location_guess, elec_progression = \
    initial_guess_axon_location(cell_ei, coords, 
        control_point_spacing=CONTROL_POINT_SPACING, 
        overshoot_length=ARRAY_OVERSHOOT_LENGTH
        )
plt.savefig(os.path.join(cell_folder, 'plots', f'axon_initial_location.png'))
plt.close(axes[0].figure)

# Use AIS electrode and last electrode in progression for global propagation velocity
# AIS electrode is the one with the earliest spike time
ais_elec = get_ais_electrode(cell_ei, amp_cutoff_mV=0.02)
last_elec = elec_progression[-1]  # Last electrode in progression

# Only use AIS if it's different from the last electrode
if ais_elec != last_elec and ais_elec != -1:
    first_elec = ais_elec
    print(f"Using AIS electrode ({ais_elec}) and last electrode ({last_elec}) for velocity pair", flush=True)
else:
    # Fallback to first electrode (soma) if AIS is same as last or not found
    first_elec = elec_progression[0]  # First electrode (soma)
    print(f"AIS electrode ({ais_elec}) same as last or not found, using soma ({first_elec}) and last ({last_elec})", flush=True)

# Ensure velocity pair is ordered correctly (i < j required by velocity calculation)
# The velocity calculation requires the first electrode index to be less than the second
if first_elec < last_elec:
    velocity_elec_i = first_elec
    velocity_elec_j = last_elec
else:
    # Swap if needed to ensure i < j
    velocity_elec_i = last_elec
    velocity_elec_j = first_elec
    print(f"Swapped velocity pair to ensure i < j: ({first_elec}, {last_elec}) -> ({velocity_elec_i}, {velocity_elec_j})", flush=True)

velocity_elecs_pairs = jnp.array([velocity_elec_i]), jnp.array([velocity_elec_j])
# Compute distance between velocity pair electrodes
velocity_pair_distance = jnp.linalg.norm(coords[velocity_elec_i] - coords[velocity_elec_j])
velocity_distance_factor = (velocity_pair_distance / 30.0) ** 2
print(f"Overrided velocity elecs pairs: single pair ({velocity_elec_i}, {velocity_elec_j})", flush=True)
print(f"left elec: {velocity_elec_i}, right elec: {velocity_elec_j}", flush=True)
print(f"Distance between velocity pair electrodes: {velocity_pair_distance:.2f} μm", flush=True)
print(f"Velocity weight adjustment factor (distance/30)^2: {velocity_distance_factor:.4f}", flush=True)


eap_plot_elecs = elec_progression[::3].tolist()
if elec_progression[-1] not in eap_plot_elecs:
    eap_plot_elecs.append(elec_progression[-1])

axes[0].plot(axon_control_points_raw[:, 0], axon_control_points_raw[:, 1], color='green', linewidth=2)
axes[0].plot(final_location_guess[:, 0], final_location_guess[:, 1], color='red', linewidth=2)
axes[0].scatter(axon_control_points[:, 0], axon_control_points[:, 1], color='blue', s=100)

#print(axon_control_points)
uncut_axon_length = jnp.sum(jnp.linalg.norm(final_location_guess[1:] - final_location_guess[:-1], axis=1))
total_axon_length = (uncut_axon_length - ARRAY_OVERSHOOT_LENGTH/2)
total_axon_length = AXON_COMP_LENGTH * round(total_axon_length/AXON_COMP_LENGTH)

if MODE == 'triplet_og':
    important_elecs = triplet_stim_elecs.tolist()
elif MODE == 'single_elec':
    important_elecs = train_test_ordering.tolist()

RGC_kwargs = {
    'electrode_locations_um': coords,
    'num_control_points': axon_control_points.shape[0],
    'TIME_STEP': 10e-3,
    'total_axon_length': total_axon_length,
    'axon_comp_length': AXON_COMP_LENGTH, 
    'control_point_spacing': CONTROL_POINT_SPACING,
    'TOTAL_SIM_TIME': 2700e-3,
    'important_elecs': important_elecs,
    'timing_features_min_amp_mV': TIMING_FEATURES_MIN_AMP_mV
}
cell_obj = RGC(**RGC_kwargs)

print(f"computing features of gt ei", flush=True)
# Use electrode progression for sodium-potassium width, use last electrode for diffusion-sodium width
# Convert to JAX arrays
elec_progression_jax = jnp.array(elec_progression)
last_elec_for_di_so = jnp.array([elec_progression[-1]], dtype=jnp.int32)  # Use last electrode to avoid empty array issues
cell_obj.set_gt_ei_features(cell_ei_100kHz, 
                            velocity_elecs_override=velocity_elecs_pairs,
                            ei_widths_so_po_elecs_override=elec_progression_jax,
                            ei_widths_di_so_elecs_override=last_elec_for_di_so)
print(f"gt features computed", flush=True)

current_params = cell_obj.get_default_kish_2023_values_dict()
current_params['axon_spline_control_points_xy'] = axon_control_points[:, :2]
current_params['axon_spline_control_points_z'] = cell_obj.get_initial_guess_z_v1()

opt_params = cell_obj.inverse_sigmoid_params_v1(current_params)

#print(f"beginning JIT compilation of gradient computation")
#(loss, aux), gradient = cell_obj.jitted_grad(opt_params)


def get_threshold_crit_voltage_estimates(cell_obj, full_params, stim_elecs, 
            MAX_CURRENT=20.0, NUM_ITERATIONS=15):
    pos_thresh, neg_thresh = [], []
    pos_critical_voltage, neg_critical_voltage = [], []
    for elec in stim_elecs:
        (lower, upper), (amp_lower, amp_upper) = cell_obj.jitted_binary_search_threshold(
            elec, MAX_CURRENT, NUM_ITERATIONS, full_params, is_positive=True)
        pos_thresh.append((lower + upper) / 2.0)
        pos_critical_voltage.append((amp_lower + amp_upper) / 2.0)

        (lower, upper), (amp_lower, amp_upper) = cell_obj.jitted_binary_search_threshold(
            elec, MAX_CURRENT, NUM_ITERATIONS, full_params, is_positive=False)
        neg_thresh.append((lower + upper) / 2.0)
        neg_critical_voltage.append((amp_lower + amp_upper) / 2.0)

    pos_thresh, neg_thresh = jnp.array(pos_thresh), jnp.array(neg_thresh)
    pos_critical_voltage, neg_critical_voltage = jnp.array(pos_critical_voltage), jnp.array(neg_critical_voltage)

    return pos_thresh, neg_thresh, pos_critical_voltage, neg_critical_voltage


def generate_estim_data(cell_obj, full_params, stim_elecs, gt_pos_thresh, gt_neg_thresh, 
            WINDOW_FACTOR=0.3, MAX_CURRENT=20.0, NUM_ITERATIONS=15, NUM_TEST_AMPS=5):

    pos_thresh, neg_thresh, pos_critical_voltage, neg_critical_voltage = \
        get_threshold_crit_voltage_estimates(cell_obj, full_params, stim_elecs, 
            MAX_CURRENT=MAX_CURRENT, NUM_ITERATIONS=NUM_ITERATIONS)

    # Assert all absolute threshold values are less than 90% of MAX_CURRENT
    # assert jnp.all(jnp.abs(pos_thresh) < 0.9 * MAX_CURRENT), \
    #     f"Some positive thresholds exceed 90% of MAX_CURRENT ({MAX_CURRENT}μA), pos_thresh: {pos_thresh}"
    # assert jnp.all(jnp.abs(neg_thresh) < 0.9 * MAX_CURRENT), \
    #     f"Some negative thresholds exceed 90% of MAX_CURRENT ({MAX_CURRENT}μA), neg_thresh: {neg_thresh}"
    
    # Only check ground truth thresholds if they are not None
    if gt_pos_thresh is not None:
        assert jnp.all(jnp.abs(gt_pos_thresh) < 0.9 * MAX_CURRENT), \
            f"Some ground truth positive thresholds exceed 90% of MAX_CURRENT ({MAX_CURRENT}μA)"
    if gt_neg_thresh is not None:
        assert jnp.all(jnp.abs(gt_neg_thresh) < 0.9 * MAX_CURRENT), \
            f"Some ground truth negative thresholds exceed 90% of MAX_CURRENT ({MAX_CURRENT}μA)"

    #we should use only the current threshold values
    #just do num test amps from 50% of gt thresholds to 150% of gt thresholds, 
    pos_test_amps = jnp.linspace(
        pos_thresh - WINDOW_FACTOR * pos_thresh,  # Ensure > 0
        pos_thresh + WINDOW_FACTOR * pos_thresh,
        NUM_TEST_AMPS
    )
    neg_test_amps = jnp.linspace(
        neg_thresh - WINDOW_FACTOR * jnp.abs(neg_thresh),
        neg_thresh + WINDOW_FACTOR * jnp.abs(neg_thresh),
        NUM_TEST_AMPS
    )

    estim_data = {
        'elecs':[],
        'intensities':[],
        'critical_thresholds':[],
        'gt_probs':[],
    }
    
    for meta_ind, elec in enumerate(stim_elecs):
        # Only add positive threshold data if gt_pos_thresh is not None
        if gt_pos_thresh is not None:
            for intensity in pos_test_amps[:, meta_ind]:
                estim_data['elecs'].append(elec)
                estim_data['intensities'].append(intensity)
                estim_data['critical_thresholds'].append(pos_critical_voltage[elec])
                gt_prob = 1.0 if intensity > gt_pos_thresh[meta_ind] else 0.0
                estim_data['gt_probs'].append(gt_prob)
        
        # Only add negative threshold data if gt_neg_thresh is not None
        if gt_neg_thresh is not None:
            for intensity in neg_test_amps[:, meta_ind]:
                estim_data['elecs'].append(elec)
                estim_data['intensities'].append(intensity)
                estim_data['critical_thresholds'].append(neg_critical_voltage[elec])
                gt_prob = 1.0 if intensity < gt_neg_thresh[meta_ind] else 0.0
                estim_data['gt_probs'].append(gt_prob)

    for key in estim_data.keys():
        estim_data[key] = jnp.array(estim_data[key])

    return estim_data

def plot_bielec_probs(gt_pos_thresh, gt_neg_thresh, full_params, cell_obj, triplet_stim_elecs, coords, amps_gsort, probs_multisite, epoch=0):
    
    pos_thresh, neg_thresh, _, _ = get_threshold_crit_voltage_estimates(cell_obj, full_params, triplet_stim_elecs)

    # Copy full_params into triplet_full_params
    triplet_full_params = {k: v for k, v in full_params.items()}
    #this can be done with vectorization
    triplet_full_params['positive_current_adjustment'] = \
        triplet_full_params['positive_current_adjustment'].at[triplet_stim_elecs].set(
            full_params['positive_current_adjustment'][triplet_stim_elecs] * (pos_thresh / gt_pos_thresh)
            )
    triplet_full_params['negative_current_adjustment'] = \
        triplet_full_params['negative_current_adjustment'].at[triplet_stim_elecs].set(
            full_params['negative_current_adjustment'][triplet_stim_elecs] * (neg_thresh / gt_neg_thresh)
            )

    _amps = jnp.linspace(jnp.min(amps_gsort), jnp.max(amps_gsort), 20)
    _amps_1, _amps_2 = jnp.meshgrid(_amps, _amps)
    bielec_amps = jnp.stack([_amps_1.ravel(), _amps_2.ravel()], axis=1)

    print('starting bielec inference', flush=True)
    elec_pairs = [[triplet_stim_elecs[1], triplet_stim_elecs[2]], \
                [triplet_stim_elecs[0], triplet_stim_elecs[2]], \
                    [triplet_stim_elecs[0], triplet_stim_elecs[1]]]

    bielec_probs = []
    for elec_pair in elec_pairs:
        bielec_probs.append(cell_obj.jitted_multielec_inference_vmapped(triplet_full_params, jnp.array(elec_pair), bielec_amps))
        bielec_probs[-1].block_until_ready()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Create title showing the current adjustments
    pos_adj = triplet_full_params['positive_current_adjustment'][triplet_stim_elecs]
    neg_adj = triplet_full_params['negative_current_adjustment'][triplet_stim_elecs]


    #scatter the compartment locations and the triplet stim elecs
    x_center = coords[triplet_stim_elecs, 0].mean()
    y_center = coords[triplet_stim_elecs, 1].mean()
    axes[0, 0].set_xlim(x_center - 75, x_center + 75)
    axes[0, 0].set_ylim(y_center - 75, y_center + 75)
    axes[0, 0].plot(full_params['compartment_locations'][:,0], full_params['compartment_locations'][:,1], marker='o', c='b', label='Cell Compartments')
    axes[0, 0].scatter(coords[triplet_stim_elecs,0], coords[triplet_stim_elecs,1], c='r', s=20)
    axes[0, 0].scatter(coords[:,0], coords[:,1], c='black', s=5)
    axes[0, 0].legend()
    # Add electrode number annotations
    for i, elec in enumerate(triplet_stim_elecs):
        axes[0, 0].annotate('Elec' + str(i+1), (coords[elec,0], coords[elec,1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    color='red', fontsize=12)

    axes[0, 1].scatter(bielec_amps[:, 0], bielec_amps[:, 1], c=bielec_probs[0], cmap='viridis', alpha=0.6, s=30)
    axes[0, 1].set_xlabel('Current Electrode 2 (μA)', fontsize=12)
    axes[0, 1].set_ylabel('Current Electrode 3 (μA)', fontsize=12)
    axes[0, 1].set_title('Model Predictions Elec1=0', fontsize=14)
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Response Probability')

    axes[0, 2].scatter(bielec_amps[:, 0], bielec_amps[:, 1], c=bielec_probs[1], cmap='viridis', alpha=0.6, s=30)
    axes[0, 2].set_xlabel('Current Electrode 1 (μA)', fontsize=12)
    axes[0, 2].set_ylabel('Current Electrode 3 (μA)', fontsize=12)
    axes[0, 2].set_title('Model Predictions Elec2=0', fontsize=14)
    plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2], label='Response Probability')

    axes[0, 3].scatter(bielec_amps[:, 0], bielec_amps[:, 1], c=bielec_probs[2], cmap='viridis', alpha=0.6, s=30)
    axes[0, 3].set_xlabel('Current Electrode 1 (μA)', fontsize=12)
    axes[0, 3].set_ylabel('Current Electrode 2 (μA)', fontsize=12)
    axes[0, 3].set_title('Model Predictions Elec3=0', fontsize=14)
    plt.colorbar(axes[0, 3].collections[0], ax=axes[0, 3], label='Response Probability')


    multisite_axes = (axes[1, 1], axes[1, 2], axes[1, 3])
    plot_stimulation_cross_sections(amps_gsort, jnp.where(probs_multisite > 0.5, 1.0, 0.0), given_axes=multisite_axes, margin=0.1)

    axes[1, 1].set_xlabel('Current Electrode 2 (μA)', fontsize=12)
    axes[1, 1].set_ylabel('Current Electrode 3 (μA)', fontsize=12)
    axes[1, 1].set_title('Experimental Data Elec1=0', fontsize=14)
    axes[1, 2].set_xlabel('Current Electrode 1 (μA)', fontsize=12)
    axes[1, 2].set_ylabel('Current Electrode 3 (μA)', fontsize=12)
    axes[1, 2].set_title('Experimental Data Elec2=0', fontsize=14)
    axes[1, 3].set_xlabel('Current Electrode 1 (μA)', fontsize=12)
    axes[1, 3].set_ylabel('Current Electrode 2 (μA)', fontsize=12)
    axes[1, 3].set_title('Experimental Data Elec3=0', fontsize=14)

    correction_info = f"Pos Adj: [{pos_adj[0]:.2f}, {pos_adj[1]:.2f}, {pos_adj[2]:.2f}]\nNeg Adj: [{neg_adj[0]:.2f}, {neg_adj[1]:.2f}, {neg_adj[2]:.2f}]"
    print(correction_info, flush=True)
    title = 'Multielectrode Stimulation Predictions vs Experimental Data'
    fig.suptitle(title, fontsize=14)

    # Add correction info text box in bottom left subplot
    axes[1, 0].text(0.05, 0.05, correction_info, 
                    transform=axes[1, 0].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    fontsize=10,
                    verticalalignment='bottom')

    plt.savefig(os.path.join(cell_folder, 'plots', f'bielec_probs_epoch_{epoch:04d}.png'))
    plt.close(fig)
    
    # Save bielec data
    bielec_data = {
        'bielec_probs': bielec_probs,
        'bielec_amps': bielec_amps,
        'pos_adj': pos_adj,
        'neg_adj': neg_adj,
        'correction_info': correction_info,
        'epoch': epoch
    }
    with open(os.path.join(cell_folder, 'data', f'bielec_data_epoch_{epoch:04d}.pkl'), 'wb') as f:
        pickle.dump(bielec_data, f)

    return fig, axes



def plot_training_progress(parameters_over_time, loss_history, epoch_losses, cell_obj, opt_params, aux, elec_coords, gt_ei, plot_elecs:list[int], epoch=0):
    """
    Plot training progress including parameters, losses, and compartment locations over time.
    
    Args:
        parameters_over_time: Dictionary of parameter values over time
        loss_history: Dictionary of individual loss values over time
        epoch_losses: List of total loss values over time
        cell_obj: RGC model object
        opt_params: Current optimization parameters
        aux: Auxiliary values from loss computation
        plot_period: How often to plot
    """
    import matplotlib.pyplot as plt
    
    # Create main figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1 (top left): All feature losses over time (excluding penalties)
    axes[0, 0].plot(epoch_losses, label='Total Loss', linewidth=2)
    # Only plot feature losses, exclude penalties
    penalty_names = set(cell_obj.penalty_weights.keys())
    for loss_name, loss_values in loss_history.items():
        if len(loss_values) > 0 and loss_name not in penalty_names:
            axes[0, 0].plot(loss_values, label=loss_name)
    axes[0, 0].set_title('Feature Losses Over Time (No Penalties)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2 (top right): Predicted EAPs
    predicted_ei = aux['eap']
    for elec in plot_elecs:
        color = f'C{elec}'  # Use matplotlib's default color cycle
        axes[0, 1].plot(predicted_ei[:, elec], color=color, linestyle='-', label=f'Predicted E{elec}')
        axes[0, 1].plot(gt_ei[:, elec], color=color, linestyle=':', label=f'Ground Truth E{elec}')
    axes[0, 1].set_title('Predicted vs Ground Truth EAPs')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Potential (mV)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3 (bottom left): Compartment locations X vs Y with electrodes
    current_params = cell_obj.forward_sigmoid_params_v1(opt_params)
    full_params = cell_obj.expand_params_v1(current_params)
    compartment_locations = full_params['compartment_locations']
    
    # X vs Y plot - first plot static EI
    gt_ei_max_values = np.max(np.abs(gt_ei), axis=0)
    axes[1, 0].scatter(elec_coords[:, 0], elec_coords[:, 1], s=1, c='black', alpha=0.3)
    axes[1, 0].scatter(elec_coords[:, 0], elec_coords[:, 1], s=gt_ei_max_values * 600.0, c='red', alpha=0.5, label='Ground Truth EI')
    
    # Then plot compartment locations
    axes[1, 0].scatter(compartment_locations[:, 0], compartment_locations[:, 1], c='blue', s=10, alpha=0.7, label='Compartments')
    axes[1, 0].set_xlabel('X (μm)')
    axes[1, 0].set_ylabel('Y (μm)') 
    axes[1, 0].set_title('X vs Y Compartment Locations')
    
    # Plot control points
    control_points_xy = current_params['axon_spline_control_points_xy']
    control_points_z = current_params['axon_spline_control_points_z']
    control_points_3d = jnp.concatenate([control_points_xy, control_points_z[:, None]], axis=1)
    axes[1, 0].scatter(control_points_3d[:, 0], control_points_3d[:, 1], c='green', s=30, label='Control Points')
    
    axes[1, 0].legend()
    
    # Plot 4 (bottom right): Compartment locations Z vs Y
    axes[1, 1].scatter(compartment_locations[:, 1], compartment_locations[:, 2], c='red', s=20, alpha=0.7, label='Compartments')
    axes[1, 1].set_xlabel('Y (μm)')
    axes[1, 1].set_ylabel('Z (μm)')
    axes[1, 1].set_title('Z vs Y Compartment Locations')
    
    # Plot control points
    axes[1, 1].scatter(control_points_3d[:, 1], control_points_3d[:, 2], c='green', s=30, label='Control Points')
    
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cell_folder, 'plots', f'training_progress_epoch_{epoch:04d}.png'))
    plt.close(fig)

    loc_fig, loc_axes = fig, axes
    
    # Create separate figure for sodium velocity loss
    if 'sodium_velocity' in loss_history and len(loss_history['sodium_velocity']) > 0:
        fig_vel, ax_vel = plt.subplots(figsize=(10, 6))
        ax_vel.plot(loss_history['sodium_velocity'], label='Sodium Velocity Loss', linewidth=2, color='red')
        ax_vel.set_title('Sodium Velocity Loss Over Time')
        ax_vel.set_xlabel('Epoch')
        ax_vel.set_ylabel('Sodium Velocity Loss')
        ax_vel.legend()
        ax_vel.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cell_folder, 'plots', f'sodium_velocity_loss_epoch_{epoch:04d}.png'))
        plt.close(fig_vel)

    
    # Create separate figure for biophysical parameters
    all_biophysical_params = [
        'HH_gNa_dendrite', 'HH_gNa_soma', 'HH_gNa_AH', 'HH_gNa_SOCB', 'HH_gNa_NR', 'HH_gNa_axon',
        'HH_gK_dendrite', 'HH_gK_soma', 'HH_gK_AH', 'HH_gK_SOCB', 'HH_gK_NR', 'HH_gK_axon',
        'radius_dendrite', 'radius_soma', 'radius_AH', 'radius_NR', 'radius_axon', 'radius_scale_factor',
        'extra_resistivity_ohm_cm'
    ]
    
    #get the param bounds
    param_bounds = cell_obj.get_param_bounds_v1()

    n_params = len(all_biophysical_params)
    n_cols = 6
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    fig_params, axes_params = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
    if n_rows == 1:
        axes_params = axes_params.reshape(1, -1)
    


    for i, param_name in enumerate(all_biophysical_params):
        row = i // n_cols
        col = i % n_cols
        
        # Convert to numpy for plotting
        param_values = [float(val) for val in parameters_over_time[param_name]]
        axes_params[row, col].plot(param_values)
        axes_params[row, col].set_title(param_name)
        axes_params[row, col].set_xlabel('Epoch')
        axes_params[row, col].set_ylabel('Value')
        axes_params[row, col].grid(True, alpha=0.3)

        bounds = param_bounds[param_name]
        # Add horizontal dotted lines for bounds
        axes_params[row, col].axhline(y=bounds[0], color='r', linestyle=':', alpha=0.5)
        axes_params[row, col].axhline(y=bounds[1], color='r', linestyle=':', alpha=0.5)
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes_params[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cell_folder, 'plots', f'all_biophysical_params_over_epochs.png'))
    plt.close(fig_params)

    return loc_fig, loc_axes

def train_rgc_model(
    cell_obj,
    opt_params,
    trainable_params_list,
    learning_rate_dict,
    num_epochs=400,
    betas=(0.9, 0.999),
    plot_period=10,
    eap_plot_elecs: list[int] = [],
    enabled_loss_features: frozenset[str] = ALL_ABLATABLE_LOSS_FEATURES,
):
    """
    Train the RGC model using the given optimization parameters.
    
    Args:
        cell_obj: RGC model object
        opt_params: Initial optimization parameters
        trainable_params_list: List of parameter names to train
        learning_rate_dict: Dictionary of learning rates for each parameter
        num_epochs: Number of training epochs
        betas: Beta parameters for Adam optimizer
        plot_period: How often to plot results
        
    Returns:
        tuple: (final_params, epoch_losses, parameters_over_time, loss_history)
    """
    
    # Initialize separate optimizers for each parameter
    optimizers = {}
    opt_states = {}
    
    for param_name in trainable_params_list:
        optimizers[param_name] = optax.adam(learning_rate=learning_rate_dict[param_name], b1=betas[0], b2=betas[1])
        opt_states[param_name] = optimizers[param_name].init(opt_params[param_name])
    
    # Storage for tracking
    epoch_losses = []
    parameters_over_time = {}
    for param_name in opt_params.keys():
        parameters_over_time[param_name] = []
    
    loss_history = {}
    for loss_name in cell_obj.ei_feature_weights.keys():
        loss_history[loss_name] = []
    for penalty_name in cell_obj.penalty_weights.keys():
        loss_history[penalty_name] = []
    
    epoch_times = []
    
    def check_for_nans(opt_params, aux, gradients):
        """Check for NaN values and print debugging info."""
        print("\n" + "="*80)
        print("DETAILED NaN DEBUGGING")
        print("="*80)
        
        print("\nOptimization parameters:")
        has_nan_in_params = False
        for param_name, param_val in opt_params.items():
            try:
                if jnp.any(jnp.isnan(param_val)):
                    has_nan_in_params = True
                    nan_count = jnp.sum(jnp.isnan(param_val))
                    print(f"  ❌ NaN found in opt_params['{param_name}']: {nan_count} NaN values")
                    print(f"     Shape: {param_val.shape}, dtype: {param_val.dtype}")
                    print(f"     Sample values: {param_val.flatten()[:5]}")
            except (TypeError, ValueError):
                continue
        if not has_nan_in_params:
            print("  ✓ No NaNs in optimization parameters")
        
        print("\nAuxiliary values:")
        has_nan_in_aux = False
        for aux_name, aux_val in aux.items():
            # Skip non-array values (like dicts, lists, etc.)
            if isinstance(aux_val, (dict, list, tuple, str)):
                # Check nested dictionaries
                if isinstance(aux_val, dict):
                    for nested_key, nested_val in aux_val.items():
                        try:
                            if jnp.any(jnp.isnan(nested_val)):
                                has_nan_in_aux = True
                                nan_count = jnp.sum(jnp.isnan(nested_val))
                                print(f"  ❌ NaN found in aux['{aux_name}']['{nested_key}']: {nan_count} NaN values")
                                print(f"     Shape: {nested_val.shape if hasattr(nested_val, 'shape') else 'scalar'}")
                        except (TypeError, ValueError):
                            continue
                continue
            try:
                if jnp.any(jnp.isnan(aux_val)):
                    has_nan_in_aux = True
                    nan_count = jnp.sum(jnp.isnan(aux_val))
                    print(f"  ❌ NaN found in aux['{aux_name}']: {nan_count} NaN values")
                    print(f"     Shape: {aux_val.shape if hasattr(aux_val, 'shape') else 'scalar'}")
            except (TypeError, ValueError):
                # Skip if aux_val is not a valid array type
                continue
        if not has_nan_in_aux:
            print("  ✓ No NaNs in auxiliary values")
        
        print("\nGradients:")
        has_nan_in_grads = False
        for grad_name, grad_val in gradients.items():
            try:
                if jnp.any(jnp.isnan(grad_val)):
                    has_nan_in_grads = True
                    nan_count = jnp.sum(jnp.isnan(grad_val))
                    print(f"  ❌ NaN found in gradients['{grad_name}']: {nan_count} NaN values")
                    print(f"     Shape: {grad_val.shape}, dtype: {grad_val.dtype}")
            except (TypeError, ValueError):
                continue
        if not has_nan_in_grads:
            print("  ✓ No NaNs in gradients")
        
        print("\n" + "="*80)
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training parameters: {trainable_params_list}")
    print(f"Learning rates: {learning_rate_dict}")
    


    # No longer saving any checkpoints or accumulating data in memory during training
    # This prevents memory leaks from accumulating aux/EI/grad data

    if MODE == 'triplet_og':
        estim_training_elecs = triplet_stim_elecs
        gt_pos_thresh = jnp.array(pos_thresholds_uA)
        gt_neg_thresh = jnp.array(neg_thresholds_uA)
        print(f'Ground truth triplet elec thresholds: pos: {gt_pos_thresh}, neg: {gt_neg_thresh}')

    elif MODE == 'single_elec':
        ais_elec = jnp.argmin(jnp.min(cell_ei, axis=0))
        print(f"AIS electrode: {ais_elec}")
        num_elecs = train_test_ordering.size
        #assert train test ordering is 1D
        assert train_test_ordering.ndim == 1
        # Check AIS against full thresholds list (not filtered train_test_ordering)
        assert ais_elec in train_test_ordering_full, f"Abandoning this Cell, the AIS electrode {ais_elec} does not have a threshold, very suspicious"

        print(f"Training electrodes: {train_test_ordering}")
        num_training_elecs = int(jnp.ceil(TRAIN_FRACTION * num_elecs))
        estim_training_elecs = list(set((train_test_ordering[0:num_training_elecs].tolist()) + [int(ais_elec)]))

        print(f'num training elecs: {num_training_elecs}')
        print(f'estim training elecs: {estim_training_elecs}')


        gt_pos_thresh = None
        gt_neg_thresh = jnp.array([stim_thresholds[_elec] for _elec in estim_training_elecs])
        
        print(f'Ground truth single elec thresholds: pos: {gt_pos_thresh}, neg: {gt_neg_thresh}')

    # Compute initial loss to normalize feature weights (using non-jitted version to avoid caching old weights)
    print("\nComputing initial losses to normalize feature weights...")
    full_params = cell_obj.expand_params_v1(cell_obj.forward_sigmoid_params_v1(opt_params))
    initial_estim_data = generate_estim_data(cell_obj, full_params, estim_training_elecs, gt_pos_thresh, gt_neg_thresh)
    # Use non-jitted loss_v1 to compute initial loss (avoids jit cache issues with weight changes)
    initial_loss, initial_aux = cell_obj.loss_v1(opt_params, initial_estim_data)
    
    # Store initial_aux for optimized dump at the end (computed once, reused later)
    
    # Extract initial feature losses
    initial_feature_losses = initial_aux['feature_losses']
    print("Initial feature losses (before normalization):")
    for feature_name, loss_val in initial_feature_losses.items():
        print(f"  {feature_name}: {float(loss_val):.6e}")
    
    if 'sodium_velocity' in enabled_loss_features:
        # LOUDLY CHECK: Ensure sodium velocity loss is valid
        print("\n" + "="*80)
        print("="*80)
        print("  CHECKING SODIUM VELOCITY LOSS VALIDITY")
        print("="*80)
        print("="*80)

        velocity_pairs = cell_obj.loss_precomputes['sodium_velocity_valid_elec_pairs']
        if isinstance(velocity_pairs, tuple) and len(velocity_pairs) == 2:
            num_pairs = len(velocity_pairs[0])
            print(f"  Number of valid velocity electrode pairs: {num_pairs}", flush=True)
            if num_pairs == 0:
                print("\n" + "!"*80)
                print("!"*80)
                print("  ERROR: NO VALID VELOCITY ELECTRODE PAIRS FOUND!")
                try:
                    print(f"  Velocity pair override: ({velocity_elec_i}, {velocity_elec_j})")
                except NameError:
                    print("  Using default velocity electrode pairs from precompute")
                print("  This will cause sodium velocity loss to be 0 throughout training.")
                print("!"*80)
                print("!"*80 + "\n")
                raise ValueError("No valid velocity electrode pairs found! Check electrode indices and ensure i < j.")
        else:
            print(f"  WARNING: Unexpected velocity_pairs format: {type(velocity_pairs)}", flush=True)

        initial_sodium_velocity_loss = float(initial_feature_losses.get('sodium_velocity', 0.0))
        print(f"  Initial sodium velocity loss: {initial_sodium_velocity_loss:.6e}", flush=True)

        if initial_sodium_velocity_loss <= 0.0:
            print("\n" + "!"*80)
            print("!"*80)
            print("  ERROR: SODIUM VELOCITY LOSS IS ZERO OR NEGATIVE!")
            print(f"  Initial sodium velocity loss: {initial_sodium_velocity_loss:.6e}")
            try:
                print(f"  Velocity pair override: ({velocity_elec_i}, {velocity_elec_j})")
            except NameError:
                print("  Using default velocity electrode pairs from precompute")
            print("  This indicates the velocity calculation is not working correctly.")
            print("  Training will fail because velocity loss will remain 0 throughout.")
            print("!"*80)
            print("!"*80 + "\n")
            raise ValueError(
                f"Sodium velocity loss is {initial_sodium_velocity_loss:.6e} (must be > 0)! "
                "Check velocity electrode pair configuration."
            )

        print("  ✓ Sodium velocity loss is valid (> 0)")
        print("="*80)
        print("="*80 + "\n")
    else:
        print("\nSkipping sodium velocity validity check (--loss_features excludes sodium_velocity).\n")
    
    # Normalize weights so each feature loss starts at 10 (except di_so which we'll set to 0)
    target_initial_loss = 10.0
    normalized_weights = {}
    for feature_name in cell_obj.ei_feature_weights.keys():
        if feature_name == 'ei_widths_di_so':
            # Set di_so weight to 0 always
            normalized_weights[feature_name] = 0.0
            print(f"  {feature_name}: weight set to 0.0 (ignored)")
        else:
            initial_loss_val = float(initial_feature_losses[feature_name])
            if initial_loss_val > 0:
                # Scale weight so initial loss becomes target_initial_loss
                current_weight = cell_obj.ei_feature_weights[feature_name]
                scale_factor = target_initial_loss / initial_loss_val
                normalized_weights[feature_name] = current_weight * scale_factor
                print(f"  {feature_name}: weight scaled from {current_weight:.6e} to {normalized_weights[feature_name]:.6e} (scale factor: {scale_factor:.6f})")
            else:
                # If initial loss is 0, keep original weight
                normalized_weights[feature_name] = cell_obj.ei_feature_weights[feature_name]
                print(f"  {feature_name}: initial loss is 0, keeping original weight {normalized_weights[feature_name]:.6e}")
    
    # Update the feature weights BEFORE jitted_grad is first called in training loop
    cell_obj.ei_feature_weights = normalized_weights
    print(f"\nNormalized feature weights: {cell_obj.ei_feature_weights}")

    apply_loss_feature_ablation(cell_obj, enabled_loss_features)
    ablated = sorted(ALL_ABLATABLE_LOSS_FEATURES - enabled_loss_features)
    print(f"Enabled loss features: {sorted(enabled_loss_features)}")
    if ablated:
        print(f"Ablated loss features (zeroed, not traced in autodiff): {ablated}")
    print(f"EI feature weights after ablation: {cell_obj.ei_feature_weights}")
    print(f"Estimation loss weight after ablation: {cell_obj.estim_loss_weight}")

    # Force recompilation of jitted_grad with new weights by making a dummy call
    # This ensures the jitted function uses the normalized weights
    print("Forcing recompilation of jitted functions with normalized weights...")
    _ = cell_obj.jitted_grad(opt_params, initial_estim_data)  # Dummy call to trigger recompilation
    print("Jitted functions recompiled with normalized weights.")
    print("="*80 + "\n")

    for epoch in tqdm(range(num_epochs), desc="Training"):
        start_time = time.time()
        
        full_params = cell_obj.expand_params_v1(cell_obj.forward_sigmoid_params_v1(opt_params))

        if epoch % 10 == 0:
            estim_data = generate_estim_data(cell_obj, full_params, estim_training_elecs, gt_pos_thresh, gt_neg_thresh)
            

        # Compute loss and gradients
        (loss_val, aux), gradients = cell_obj.jitted_grad(opt_params, estim_data)
        
        # Don't store aux, EI, or gradients - they're only needed for the current iteration
        # This prevents memory accumulation during training

        # Check for NaN values
        has_nan = jnp.isnan(loss_val)
        for grad_val in gradients.values():
            if jnp.any(jnp.isnan(grad_val)):
                has_nan = True
                break
        
        if has_nan:
            print(f"NaN detected at epoch {epoch}! Running detailed debugging...")
            check_for_nans(opt_params, aux, gradients)
            raise ValueError("NaN detected in optimization")
        
        # Update parameters using separate optimizers
        for param_name in opt_params.keys():
            param_gradients = gradients[param_name]
            if param_name in trainable_params_list:
                updates, opt_states[param_name] = optimizers[param_name].update(
                    param_gradients, opt_states[param_name]
                )
                opt_params[param_name] = optax.apply_updates(
                    opt_params[param_name], updates
                )
        
        # Check for NaN values in updated parameters
        params_has_nan = False
        for param_name, param_val in opt_params.items():
            if jnp.any(jnp.isnan(param_val)):
                params_has_nan = True
        
        if params_has_nan:
            print(f"NaN detected in parameters at epoch {epoch}!")
            check_for_nans(opt_params, aux, gradients)
            raise ValueError("NaN detected in parameters")
        
        # Store results
        current_physical_params = cell_obj.forward_sigmoid_params_v1(opt_params)
        for param_name in opt_params.keys():
            parameters_over_time[param_name].append(current_physical_params[param_name])
        
        for loss_name in cell_obj.ei_feature_weights.keys():
            loss_history[loss_name].append(float(aux['feature_losses'][loss_name]))

        for penalty_name in cell_obj.penalty_weights.keys():
            loss_history[penalty_name].append(float(aux['penalty_losses'][penalty_name]))
        
        epoch_losses.append(float(loss_val))
        
        # Print progress and plot
        if epoch % plot_period == 0:
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(gradients)))
            print(f"Epoch {epoch}, Loss: {loss_val:.6f}, Grad Norm: {grad_norm:.6f}")
            
            # Plot training progress
            loc_fig, loc_axes = plot_training_progress(parameters_over_time, loss_history, epoch_losses, cell_obj, opt_params, aux, coords, cell_ei_100kHz, eap_plot_elecs, epoch)
            
            if MODE == 'triplet_og' and not SKIP_BIELEC:
                bielec_fig, bielec_axes = plot_bielec_probs(
                    gt_pos_thresh, gt_neg_thresh, full_params, cell_obj, triplet_stim_elecs, coords, amps_gsort, probs_multisite, epoch
                )
        
        end_time = time.time()
        epoch_times.append(end_time - start_time)
        
        if epoch % 50 == 0:
            avg_time = np.mean(epoch_times[-50:])
            print(f"Average epoch time (last 50): {avg_time:.2f} seconds")
    
    # Get final parameters
    final_params = cell_obj.forward_sigmoid_params_v1(opt_params)
    
    # Calculate actual number of epochs completed
    actual_epochs_completed = len(epoch_losses)
    
    # VERY LOUD display of epochs completed
    print("\n" + "="*80)
    print("="*80)
    print(f"  TRAINING COMPLETED: {actual_epochs_completed} EPOCHS ACTUALLY RUN")
    print(f"  TARGET WAS: {num_epochs} EPOCHS")
    print("="*80)
    print("="*80 + "\n")
    
    print(f"Training completed! Final loss: {epoch_losses[-1]:.6f}")
    
    # Save final results
    final_results = {
        'final_params': final_params,
        'epoch_losses': epoch_losses,
        'parameters_over_time': parameters_over_time,
        'loss_history': loss_history,
        'actual_epochs_completed': actual_epochs_completed,
        'target_epochs': num_epochs,
        'cell_info': {
            'cell_index': CELL_INDEX,
            'piece': piece,
            'cell_id': cell_id,
            'cell_type': cell_type
        }
    }
    with open(os.path.join(cell_folder, 'data', 'final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    
    # Create optimized dump file with essential data only
    # Compute last aux by running one final forward pass (needed for evaluation/visualization)
    print("Creating optimized dump file...")
    print("Computing final aux for optimized dump...")
    final_full_params = cell_obj.expand_params_v1(final_params)
    final_estim_data = generate_estim_data(cell_obj, final_full_params, estim_training_elecs, gt_pos_thresh, gt_neg_thresh)
    _, final_aux = cell_obj.loss_v1(opt_params, final_estim_data)
    
    # Reuse initial_aux that was computed at the start (already computed for weight normalization)
    optimized_data = {
        'RGC_kwargs': RGC_kwargs,
        'first_aux': initial_aux,
        'last_aux': final_aux,
        'actual_epochs_completed': actual_epochs_completed,
        'target_epochs': num_epochs,
        'enabled_loss_features': sorted(enabled_loss_features),
    }
    
    optimized_path = os.path.join(cell_folder, 'data', 'optimized_dump.pkl')
    with open(optimized_path, 'wb') as f:
        pickle.dump(optimized_data, f)
    
    print(f"Optimized dump file saved to {optimized_path}")

    # Save minimal parameters for evaluation (top-level output_dir/for_evaluation)
    try:
        last_aux = optimized_data.get("last_aux")
        if last_aux is None or "full_params" not in last_aux:
            raise KeyError("optimized_data['last_aux']['full_params'] missing")
        eval_path = _save_for_evaluation_pickle(
            OUTPUT_DIR,
            "rgc_train_loss_balanced",
            int(cell_id),
            str(piece),
            RGC_kwargs,
            last_aux["full_params"],
        )
        print(f"For-evaluation pickle saved to {eval_path}")
    except Exception as e:
        print(f"WARNING: failed to write for-evaluation pickle ({e})")
    
    return final_params, epoch_losses, parameters_over_time, loss_history


if __name__ == '__main__':
    # Set loss term weightings based on mode
    if MODE == 'triplet_og':
        # Copy weightings from rgc_model.py for triplet mode
        # Adjust sodium_velocity weight by distance factor: divide by (distance/30)^2
        base_velocity_weight = 1e1 * ((cell_obj.TIME_STEP/10e-3) ** 2)
        adjusted_velocity_weight = base_velocity_weight / velocity_distance_factor
        cell_obj.ei_feature_weights = {
            'sodium_peaks': 1e5,
            'diffusion_peaks': 2e5,
            'potassium_peaks': 1e5,
            'sodium_velocity': adjusted_velocity_weight,
            'ei_widths_so_po': 1e-3,
            'ei_widths_di_so': 1e-3,
        }
        
        cell_obj.penalty_weights = {
            'first_order_continuity': 1e0 * (80.0/cell_obj.control_point_spacing) ** 2,
            'second_order_continuity': 5e-2 * (80.0/cell_obj.control_point_spacing) ** 2,
            'axon_rise': 2e-1 * (80.0/cell_obj.control_point_spacing) ** 2,
        }
        
        cell_obj.estim_loss_weight = 10.0
        
    elif MODE == 'single_elec':
        # Use same weightings as triplet mode
        # Adjust sodium_velocity weight by distance factor: divide by (distance/30)^2
        base_velocity_weight = 1e1 * ((cell_obj.TIME_STEP/10e-3) ** 2)
        adjusted_velocity_weight = base_velocity_weight / velocity_distance_factor
        cell_obj.ei_feature_weights = {
            'sodium_peaks': 1e5,
            'diffusion_peaks': 2e5,
            'potassium_peaks': 1e5,
            'sodium_velocity': adjusted_velocity_weight,
            'ei_widths_so_po': 1e-3,
            'ei_widths_di_so': 1e-3,
        }
        
        cell_obj.penalty_weights = {
            'first_order_continuity': 1e0 * (80.0/cell_obj.control_point_spacing) ** 2,
            'second_order_continuity': 5e-2 * (80.0/cell_obj.control_point_spacing) ** 2,
            'axon_rise': 2e-1 * (80.0/cell_obj.control_point_spacing) ** 2,
        }
        
        cell_obj.estim_loss_weight = 10.0
    
    print(f"Loss weightings set for mode: {MODE}")
    print(f"EI feature weights: {cell_obj.ei_feature_weights}")
    print(f"Penalty weights: {cell_obj.penalty_weights}")
    print(f"Estimation loss weight: {cell_obj.estim_loss_weight}")
    print(f"Enabled loss features: {sorted(ENABLED_LOSS_FEATURES)}")

    trainable_params_list = [
    'HH_gNa_dendrite', 'HH_gNa_soma', 'HH_gNa_AH', 'HH_gNa_SOCB', 'HH_gNa_NR', 'HH_gNa_axon',
    'HH_gK_dendrite', 'HH_gK_soma', 'HH_gK_AH', 'HH_gK_SOCB', 'HH_gK_NR', 'HH_gK_axon',
    'radius_dendrite', 'radius_soma', 'radius_AH', 'radius_NR', 'radius_axon', 'radius_scale_factor',
    'axon_spline_control_points_xy', 'axon_spline_control_points_z', 'extra_resistivity_ohm_cm'
    ]

    learning_rate_dict = {
        'HH_gNa_dendrite': 1e-2, 'HH_gNa_soma': 1e-2, 'HH_gNa_AH': 1e-2, 'HH_gNa_SOCB': 1e-2, 'HH_gNa_NR': 1e-2, 'HH_gNa_axon': 1e-2,
        'HH_gK_dendrite': 1e-2, 'HH_gK_soma': 1e-2, 'HH_gK_AH': 1e-2, 'HH_gK_SOCB': 1e-2, 'HH_gK_NR': 1e-2, 'HH_gK_axon': 1e-2,
        'radius_dendrite': 1e-2, 'radius_soma': 1e-2, 'radius_AH': 1e-2, 'radius_NR': 1e-2, 'radius_axon': 1e-2, 'radius_scale_factor': 1e-2,
        'axon_spline_control_points_xy': 1e0, 'axon_spline_control_points_z': 1e-1,  # Higher learning rate for location parameters
        'extra_resistivity_ohm_cm': 1e-2
    }

    # Run training
    print("Starting RGC model training...")
    final_params, epoch_losses, parameters_over_time, loss_history = train_rgc_model(
        cell_obj,
        opt_params,
        trainable_params_list,
        learning_rate_dict,
        num_epochs=1000,
        plot_period=10,
        eap_plot_elecs=eap_plot_elecs,
        enabled_loss_features=ENABLED_LOSS_FEATURES,
    )
