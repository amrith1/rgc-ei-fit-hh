import numpy as np
from splinex.bspline import BSpline

def axonorsomaRatio(wave, uppBound=1.6, lowBound=0.8):
    try:
        #Get index and value of (negative) min {only one}
        minind = np.argmin(wave)
        minval = np.min(wave)

        #Get max vals to either side of min
        maxvalLeft = np.max(wave[0:minind])
        maxvalRight = np.max(wave[minind:])

        if np.abs(minval) < max(maxvalLeft,maxvalRight):
            rCompt = 'dendrite'
        else:
            if maxvalRight == 0:
                ratio = 0
            else:
                ratio = maxvalLeft/maxvalRight
            if ratio > uppBound:
                rCompt = 'axon'
            elif ratio < lowBound: #FUDGED
                rCompt = 'soma'
            else:
                rCompt = 'mixed'
    except ValueError:
        rCompt = 'error' #wave is abnormally shaped (usually has min at leftmost or rightmost point)

    return rCompt

def get_max_value_electrode(ei):
    """
    Get the electrode with maximum amplitude.
    
    Args:
        ei: Extracellular potential data (time x electrodes)
        coords: Electrode coordinates (electrodes x 2)
        
    Returns:
        int: Index of the electrode with maximum amplitude
    """
    ei_max_values = np.max(np.abs(ei), axis=0)
    return np.argmax(ei_max_values)

def get_ais_electrode(ei_mV, amp_cutoff_mV=0.02):
    """
    Get the AIS electrode (earliest argmin_idx with sufficient amplitude).
    
    Args:
        ei_mV: Extracellular potential data (time x electrodes)
        amp_cutoff_mV: Amplitude cutoff for considering electrodes
        
    Returns:
        int: Index of the AIS electrode, or -1 if not found
    """
    ei_max_values = np.max(np.abs(ei_mV ), axis=0)
    argmin_idxs = [np.argmin(ei_mV[:, elec]) for elec in range(ei_mV.shape[1])]
    
    # Get indices of top 10 electrodes by amplitude
    top_elec_idxs = np.argsort(ei_max_values)[-10:]

    ais_elec = -1
    min_time = 1e9
    max_val = 0.0
    for elec in top_elec_idxs:
        if ei_max_values[elec] < amp_cutoff_mV:
            continue
        if argmin_idxs[elec] < min_time or (min_time == argmin_idxs[elec] and ei_max_values[elec] > max_val):
            min_time = argmin_idxs[elec]
            ais_elec = elec
    
    return ais_elec

def get_axon_electrodes(ei_mV, amp_cutoff_mV=0.006):
    """
    Get axon electrodes that qualify based on axonorsomaRatio and are the largest recording for their argmin.
    
    Args:
        ei_mV: Extracellular potential data (time x electrodes)
        amp_cutoff_mV: Amplitude cutoff for considering electrodes
        
    Returns:
        list: List of dictionaries with 'elec_idx', 'argmin_idx', 'max_value', and 'coords' for each axon electrode
    """
    ei_max_values = np.max(np.abs(ei_mV), axis=0)
    argmin_idxs = [np.argmin(ei_mV[:, elec]) for elec in range(ei_mV.shape[1])]
    
    # Group electrodes by type and argmin_idx
    mappings = {}
    for elec in range(ei_mV.shape[1]):
        if ei_max_values[elec] < 0.005:  # 5uV is really small
            continue
            
        elec_type = axonorsomaRatio(ei_mV[:, elec])
        # Simplify types - treat 'mixed' as axon
        if elec_type not in ['soma', 'dendrite', 'error']:
            elec_type = 'axon'
            
        summary = (elec_type, argmin_idxs[elec])
        if summary not in mappings:
            mappings[summary] = []
        mappings[summary].append(elec)
    
    # Extract axon electrodes
    axon_electrodes = []
    for summary, elecs in mappings.items():
        type, argmin_idx = summary
        if type == 'axon':
            # Find electrode with max value among this group
            max_elec = max(elecs, key=lambda e: ei_max_values[e])
            if ei_max_values[max_elec] >= amp_cutoff_mV:
                axon_electrodes.append({
                    'elec_idx': max_elec,
                    'argmin_idx': argmin_idx,
                    'max_value': ei_max_values[max_elec]
                })
    
    # Sort by argmin_idx
    axon_electrodes.sort(key=lambda x: x['argmin_idx'])
    
    elec_indices = [elec['elec_idx'] for elec in axon_electrodes]
    elec_argmin_idxs = [elec['argmin_idx'] for elec in axon_electrodes]
    elec_max_values = [elec['max_value'] for elec in axon_electrodes]

    return elec_indices, elec_argmin_idxs, elec_max_values

def filter_axon_electrodes(axon_elecs, ais_elec, soma_elec, coords, dist_thresh=150.0):
    filtered_axon_elecs = []
    for axon_elec in axon_elecs:
        dist_to_ais = jnp.linalg.norm(coords[axon_elec] - coords[ais_elec])
        dist_to_soma = jnp.linalg.norm(coords[axon_elec] - coords[soma_elec]) 
        if dist_to_ais > dist_thresh and dist_to_soma > dist_thresh:
            filtered_axon_elecs.append(axon_elec)
    return filtered_axon_elecs

import matplotlib.pyplot as plt
import jax.numpy as jnp
# Upsample EI by factor of 2 using linear interpolation
# use lancosz3 method
import jax.image
#ei_upsampled = jax.image.resize(cell_0_ei, (cell_0_ei.shape[0]*2, cell_0_ei.shape[1]), method='lanczos3')
#cell_0_ei = np.array(ei_upsampled)

#ei= cell_0_ei

def plot_ei_facts(ei, coords):

    fig, axes = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
    ax = axes[0]
    time_series_ax = axes[1]

    ei_max_values = np.max(np.abs(ei), axis=0)
    ax.scatter(coords[:, 0], coords[:, 1], s=1)

    color_dict = {'axon': 'blue', 
                'soma': 'green', 
                'dendrite': 'red', 
                'error': 'gray'}
    
    ax.scatter(coords[:, 0], coords[:, 1], s=ei_max_values * 600.0, c='black')

    max_value_elec = get_max_value_electrode(ei)
    ax.scatter(coords[max_value_elec, 0], coords[max_value_elec, 1], s=ei_max_values[max_value_elec] * 600.0, c='orange')
    time_series_ax.legend()

    ais_elec = get_ais_electrode(ei, amp_cutoff_mV=0.02)
    ax.scatter(coords[ais_elec, 0], coords[ais_elec, 1], marker='*', s=200, c='yellow', edgecolor='black', zorder=10)
    # Find electrode with smallest argmin_idx subject to the value being less than 0.01
    axon_elecs, elec_argmin_idxs, elec_max_values = get_axon_electrodes(ei)
    filtered_axon_elecs = filter_axon_electrodes(axon_elecs, ais_elec, max_value_elec, coords, dist_thresh=150.0)
    ax.scatter(coords[filtered_axon_elecs, 0], coords[filtered_axon_elecs, 1], marker='*', s=200, c='yellow', edgecolor='black', zorder=10)
    
    for elec_idx, elec_argmin_idx in zip(filtered_axon_elecs, elec_argmin_idxs):
        ax.annotate(f'{elec_argmin_idx}', (coords[elec_idx, 0], coords[elec_idx, 1]), fontsize=20, color='red')

    #plot the axon elecs on the time series
    for elec_idx, elec_argmin_idx in zip(filtered_axon_elecs, elec_argmin_idxs):
        time_series_ax.plot(ei[:, elec_idx], label=f'Electrode {elec_idx}')
    time_series_ax.legend()

    return axes

def get_evenly_spaced_point_indices(coords:jnp.ndarray, spacing_um:float, num_points:int):
    adjacent_distances = jnp.linalg.norm(coords[1:] - coords[:-1], axis=1)
    jax.lax.stop_gradient(adjacent_distances)
    cumulative_distance = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(adjacent_distances)])
    markers = jnp.linspace(0, spacing_um * num_points, num_points)
    assert markers.shape[0] == num_points
    distance_to_markers = jnp.abs(cumulative_distance[None, :] - markers[:, None])
    nearest_elec_idx = jnp.argmin(distance_to_markers, axis=1)
    assert nearest_elec_idx.shape[0] == markers.shape[0]
    return nearest_elec_idx

def initial_guess_axon_location(ei_mV, coords, output_points=5000, overshoot_length=500.0, control_point_spacing=40.0):
    axon_elecs, elec_argmin_idxs, elec_max_values = get_axon_electrodes(ei_mV)
    ais_elec = get_ais_electrode(ei_mV, amp_cutoff_mV=0.02)
    soma_elec = get_max_value_electrode(ei_mV)

    # Filter out axon electrodes that are too close to AIS or soma
    filtered_axon_elecs = filter_axon_electrodes(axon_elecs, ais_elec, soma_elec, coords, dist_thresh=150.0)

    elec_progression = np.array([soma_elec, ais_elec] + filtered_axon_elecs)
    #get the coordinates of the electrodes in the order of the elec_progression
    axon_coords = coords[elec_progression, :]

    #add on another point at the end which has the property that it lies on
    #line between soma_coord and last axon elec and it is overshoot_length away from the last axon elec
    soma_coord = axon_coords[0, :]
    last_axon_coord = axon_coords[-1, :]
    overshoot_point = last_axon_coord + \
        (last_axon_coord - soma_coord) * overshoot_length / jnp.linalg.norm(last_axon_coord - soma_coord)
    axon_coords = jnp.concatenate([axon_coords, jnp.array([overshoot_point])], axis=0)


    interpolator = BSpline(n_in=axon_coords.shape[0], n_out=output_points, degree=3)
    axon_control_points_raw, _, _ = interpolator(axon_coords)

    adjacent_distances = jnp.linalg.norm(axon_control_points_raw[1:] - axon_control_points_raw[:-1], axis=1)
    cumulative_distance = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(adjacent_distances)])
    num_points = jnp.floor(cumulative_distance[-1] / control_point_spacing).astype(jnp.int32)
    
    evenly_spaced_points = get_evenly_spaced_point_indices(axon_control_points_raw, spacing_um=control_point_spacing, num_points=num_points)
    axon_control_points = axon_control_points_raw[evenly_spaced_points, :]

    follow_interpolator = BSpline(n_in=axon_control_points.shape[0], n_out=output_points, degree=3)
    final_location_guess, _, _ = follow_interpolator(axon_control_points)

    return axon_control_points, axon_control_points_raw, final_location_guess, elec_progression