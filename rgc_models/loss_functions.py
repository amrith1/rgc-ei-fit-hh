from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.linalg


"""
The EI is a 2D array of shape (time, electrodes)
"""

def first_order_continuity_loss(compartment_locations:jnp.ndarray, compartment_spacing:float=1.0):
    """
    Punish distance between adjacent compartments deviating from the expected compartment spacing
    """
    # Assert compartment_locations is 2D with second dimension of size 3
    assert len(compartment_locations.shape) == 2, "compartment_locations must be 2D array"

    lengths = jnp.linalg.norm(compartment_locations[1:, :] - compartment_locations[:-1, :], axis=1)
    return jnp.mean(jnp.square(lengths - compartment_spacing))

def second_order_continuity_loss(compartment_locations:jnp.ndarray):
    """
    Punish second order changes in the distance between adjacent compartments
    """
    first_derivatives = jnp.diff(compartment_locations, axis=0)
    second_derivatives = jnp.diff(first_derivatives, axis=0)
    return jnp.mean(jnp.square(second_derivatives))

def third_order_continuity_loss(compartment_locations:jnp.ndarray):
    """
    Punish third order changes in the distance between adjacent compartments
    """
    first_derivatives = jnp.diff(compartment_locations, axis=0)
    second_derivatives = jnp.diff(first_derivatives, axis=0)
    third_derivatives = jnp.diff(second_derivatives, axis=0)
    return jnp.mean(jnp.square(third_derivatives))

def global_curvature_loss(compartment_locations:jnp.ndarray, gamma:float=0.0, compartment_spacing:float=1.0):
    """
    Generate a term which gets larger as the cable deviates from an evenly spaced straight line..
    """
    # Assert compartment_locations is 2D with second dimension of size 3
    assert len(compartment_locations.shape) == 2, "compartment_locations must be 2D array"
    assert compartment_locations.shape[1] == 3, "compartment_locations second dimension must have size 3"

    # Use squared distances to avoid numerical issues with sqrt
    diff_vectors = compartment_locations[:, None, :] - compartment_locations[None, :, :]
    pairwise_dist_squared = jnp.sum(jnp.square(diff_vectors), axis=2)
    
    sequence = jnp.arange(compartment_locations.shape[0], dtype=jnp.float32) * compartment_spacing
    no_curve_dist_squared = jnp.square(jnp.abs(sequence[None, :] - sequence[:, None]))
    # penalize closer distances more than futher to allow global curvature
    # if gamma is 0, then the penalty is 1 for all distances

    weight_decay = jnp.exp(-gamma * compartment_spacing * jnp.sqrt(no_curve_dist_squared))

    diff = pairwise_dist_squared - no_curve_dist_squared
    
    # Use smooth L1 loss to avoid double squaring
    epsilon = 1e-8
    return jnp.mean(jnp.sqrt(jnp.square(diff) + epsilon) * weight_decay)


if __name__ == "__main__":
    # Test bench for curvature_loss gradient computation
    print("Testing curvature_loss gradient computation...")
    
    # Test with perfectly straight line (should cause gradient issues)
    n_compartments = 10
    straight_line = jnp.arange(n_compartments, dtype=jnp.float32)[:, None] * jnp.array([1.0, 0.0, 0.0])
    
    print(f"Input shape: {straight_line.shape}")
    print(f"Input values: {straight_line}")
    
    try:
        # Test forward pass
        loss_value = global_curvature_loss(straight_line)
        print(f"Forward pass OK: loss = {loss_value}")
        
        # Test gradient computation
        grad_fn = jax.grad(global_curvature_loss)
        gradient = grad_fn(straight_line)
        print(f"Gradient computation OK: gradient shape = {gradient.shape}")
        print(f"Gradient values: {gradient}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with slightly curved line
    curved_line = straight_line + 0.1 * jnp.sin(jnp.arange(n_compartments, dtype=jnp.float32))[:, None] * jnp.array([0.0, 1.0, 0.0])
    
    print(f"\nTesting with curved line...")
    print(f"Curved input values: {curved_line}")
    
    try:
        loss_value = global_curvature_loss(curved_line)
        print(f"Forward pass OK: loss = {loss_value}")
        
        gradient = grad_fn(curved_line)
        print(f"Gradient computation OK: gradient shape = {gradient.shape}")
        print(f"Gradient values: {gradient}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def soft_argmax(pdf: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """
    Compute expected index (soft argmax) along an axis.
    Given a probability distribution, returns the expected value of the index.
    """
    indices = jnp.arange(pdf.shape[axis], dtype=jnp.float32)
    # Reshape indices to broadcast correctly
    shape = [1] * pdf.ndim
    shape[axis] = pdf.shape[axis]
    indices = indices.reshape(shape)
    return jnp.sum(indices * pdf, axis=axis)


def upsample_ei(ei:jnp.ndarray, factor:int) -> jnp.ndarray:
    """
    Upsample the EI by a factor of 2
    """
    return jax.image.resize(ei, (ei.shape[0]*factor, ei.shape[1]), 'bilinear')

def sodium_peaks(ei:jnp.ndarray) -> jnp.ndarray:
    return jnp.min(ei, axis=0)

def diffusion_peaks(ei: jnp.ndarray) -> jnp.ndarray:
    """
    Find the maximum value that occurs before the minimum value for each electrode.
    AKA capacitive peaks
    """
    # Get the minimum indices for each electrode
    min_indices = jnp.argmin(ei, axis=0)
    # Create a mask for values before the minimum
    time_indices = jnp.arange(ei.shape[0])[:, None]
    before_min_mask = time_indices < min_indices
    # Replace values after minimum with 0.0 using where
    # diffusion peaks must be non-negative
    masked_ei = jnp.where(before_min_mask, ei, 0.0)
    # Get max of masked values
    max_before_min = jnp.max(masked_ei, axis=0)
    return max_before_min

def potassium_peaks(ei:jnp.ndarray) -> jnp.ndarray:
    """
    Find the maximum value that occurs after the minimum value for each electrode.
    """
    min_indices = jnp.argmin(ei, axis=0)
    # Create a mask for values after the minimum
    time_indices = jnp.arange(ei.shape[0])[:, None]
    after_min_mask = time_indices > min_indices
    # Replace values before minimum 0.0
    # potassium peaks must be non-negative
    masked_ei = jnp.where(after_min_mask, ei, 0.0)
    # Get max of masked values
    max_after_min = jnp.max(masked_ei, axis=0)
    return max_after_min

def get_masked_eis(ei:jnp.ndarray, mask_value:float=0.0) -> jnp.ndarray:
    """
    isolates the peaks of the EIs so softmax application returns a bell curve
    """

    #set 0 all data left of the diffusion peak and right of the potassium peak
    
    sodium_indices = jnp.argmin(ei, axis=0)

    #now calculate the diffusion indices
    time_indices = jnp.arange(ei.shape[0])[:, None]

    before_sodium_mask = time_indices < sodium_indices
    after_sodium_mask = time_indices > sodium_indices

    potassium_ei = jnp.where(before_sodium_mask, mask_value, ei)
    diffusion_ei = jnp.where(after_sodium_mask, mask_value, ei)

    potassium_indices = jnp.argmax(potassium_ei, axis=0)
    diffusion_indices = jnp.argmax(diffusion_ei, axis=0)

    before_diffusion_mask = time_indices < diffusion_indices
    after_potassium_mask = time_indices > potassium_indices

    _int_sodium_ei = jnp.where(before_diffusion_mask, mask_value, ei)
    sodium_ei = jnp.where(after_potassium_mask, mask_value, _int_sodium_ei)

    return sodium_ei, diffusion_ei, potassium_ei

def precompute_valid_pt_elecs(ei:jnp.ndarray, min_req_value_mV:float=0.0, component:str='sodium', additional_mask=None) -> jnp.ndarray:
    sodium_ei, diffusion_ei, potassium_ei = get_masked_eis(ei, mask_value=0.0)
    if component == 'sodium':
        sodium_peaks = jnp.abs(jnp.min(sodium_ei, axis=0))
        elec_valid_bool= jax.lax.stop_gradient(sodium_peaks > min_req_value_mV)
    elif component == 'diffusion':
        diffusion_peaks = jnp.max(diffusion_ei, axis=0)
        elec_valid_bool = jax.lax.stop_gradient(diffusion_peaks > min_req_value_mV)
    elif component == 'potassium':
        potassium_peaks = jnp.max(potassium_ei, axis=0)
        elec_valid_bool = jax.lax.stop_gradient(potassium_peaks > min_req_value_mV)
    else:
        raise ValueError(f"Invalid component: {component}")
    
    elec_pairwise_valid_bool = jnp.zeros((ei.shape[1], ei.shape[1]))
    elec_indices = jnp.arange(ei.shape[1])

    elec_pairwise_valid_bool = (elec_valid_bool[None, :] & elec_valid_bool[:, None]) \
        & (elec_indices[:, None] < elec_indices[None, :]) #i < j
    
    if additional_mask is not None:
        elec_pairwise_valid_bool = elec_pairwise_valid_bool & additional_mask
    
    indices_pairs = jnp.argwhere(elec_pairwise_valid_bool)
    return indices_pairs[:, 0], indices_pairs[:, 1]

def pairwise_time_differences(ei:jnp.ndarray, peak_value:float=10.0, 
        min_req_value_mV:float=0.0, component:str='sodium', 
        precomputed_valid_elec_i_indices:jnp.ndarray=None,
        precomputed_valid_elec_j_indices:jnp.ndarray=None
        ) -> jnp.ndarray:
    """
    Compute pairwise time differences between electrode peaks using soft argmax.
    
    Instead of cross-correlation, we normalize the EI to a PDF and compute the
    expected time index (soft argmax) for each electrode, then take differences.
    """
    sodium_ei, diffusion_ei, potassium_ei = get_masked_eis(ei, mask_value=0.0)

    if component == 'sodium':
        sodium_peaks = jnp.abs(jnp.min(sodium_ei, axis=0))
        elec_valid_bool = jax.lax.stop_gradient(sodium_peaks > min_req_value_mV)
        betas = jnp.where(elec_valid_bool, peak_value / jnp.min(sodium_ei, axis=0), 0.0)
        normalized_ei = jax.nn.softmax(betas[None, :] * sodium_ei, axis=0)
    elif component == 'diffusion':
        diffusion_peaks = jnp.max(diffusion_ei, axis=0)
        elec_valid_bool = jax.lax.stop_gradient(diffusion_peaks > min_req_value_mV)
        betas = jnp.where(elec_valid_bool, peak_value / jnp.max(diffusion_ei, axis=0), 0.0)
        normalized_ei = jax.nn.softmax(betas[None, :] * diffusion_ei, axis=0)
    elif component == 'potassium':
        potassium_peaks = jnp.max(potassium_ei, axis=0)
        elec_valid_bool = jax.lax.stop_gradient(potassium_peaks > min_req_value_mV)
        betas = jnp.where(elec_valid_bool, peak_value / jnp.max(potassium_ei, axis=0), 0.0)
        normalized_ei = jax.nn.softmax(betas[None, :] * potassium_ei, axis=0)

    num_electrodes = ei.shape[1]
    
    # Compute soft argmax (expected time index) for each electrode
    peak_times = soft_argmax(normalized_ei, axis=0)  # shape: (n_electrodes,)
    
    # Compute pairwise differences: peak_times[i] - peak_times[j]
    pairwise_diffs = peak_times[:, None] - peak_times[None, :]
    
    elec_indices = jnp.arange(num_electrodes)
    
    # If override pairs are provided, use them directly (bypass amplitude checks)
    if precomputed_valid_elec_i_indices is not None and precomputed_valid_elec_j_indices is not None:
        precomputed_mask = jnp.zeros((num_electrodes, num_electrodes), dtype=jnp.bool_)
        precomputed_mask = precomputed_mask.at[precomputed_valid_elec_i_indices, precomputed_valid_elec_j_indices].set(True)
        # Override: use only the precomputed pairs, ignore amplitude checks
        valid_pairs_bool = precomputed_mask & (elec_indices[:, None] < elec_indices[None, :])
    else:
        # Default: Create valid pairs mask (upper triangular, both electrodes valid based on amplitude)
        valid_pairs_bool = (elec_valid_bool[:, None] & elec_valid_bool[None, :]) & \
                           (elec_indices[:, None] < elec_indices[None, :])
    
    # Zero out invalid pairs
    pairwise_diffs = jnp.where(valid_pairs_bool, pairwise_diffs, 0.0)

    return pairwise_diffs, normalized_ei, valid_pairs_bool

def precompute_valid_width_elecs(ei:jnp.ndarray, min_req_value_mV:float=0.0, component_one:str='sodium', component_two:str='potassium') -> jnp.ndarray:
    sodium_ei, diffusion_ei, potassium_ei = get_masked_eis(ei, mask_value=0.0)

    sodium_peaks = jnp.abs(jnp.min(sodium_ei, axis=0))
    sodium_elec_valid_bool= sodium_peaks > min_req_value_mV
    potassium_peaks = jnp.max(potassium_ei, axis=0)
    potassium_elec_valid_bool = potassium_peaks > min_req_value_mV
    diffusion_peaks = jnp.max(diffusion_ei, axis=0)
    diffusion_elec_valid_bool = diffusion_peaks > min_req_value_mV

    # Get indices of electrodes that have valid sodium and potassium peaks
    valid_so_po_indices = jnp.where(sodium_elec_valid_bool & potassium_elec_valid_bool)[0]
    valid_di_po_indices = jnp.where(potassium_elec_valid_bool & diffusion_elec_valid_bool)[0]
    valid_di_so_indices = jnp.where(diffusion_elec_valid_bool & sodium_elec_valid_bool)[0]

    if component_one == 'sodium' and component_two == 'potassium':
        return valid_so_po_indices
    elif component_one == 'diffusion' and component_two == 'potassium':
        return valid_di_po_indices
    elif component_one == 'diffusion' and component_two == 'sodium':
        return valid_di_so_indices
    else:
        raise ValueError(f"Invalid component combination: {component_one} and {component_two}")

def ei_widths(ei:jnp.ndarray, peak_value:float=10.0, component_one:str='sodium', 
              component_two:str='potassium', min_req_value_mV:float=0.0, precomputed_valid_elecs:jnp.ndarray=None) -> jnp.ndarray:
    """
    Compute the time difference (width) between two components on each electrode using soft argmax.
    
    Instead of cross-correlation, we compute the expected time index for each component
    and take the difference.
    """
    sodium_ei, diffusion_ei, potassium_ei = get_masked_eis(ei, mask_value=0.0)

    sodium_peaks = jnp.abs(jnp.min(sodium_ei, axis=0))
    sodium_elec_valid_bool = jax.lax.stop_gradient(sodium_peaks > min_req_value_mV)
    potassium_peaks = jnp.max(potassium_ei, axis=0)
    potassium_elec_valid_bool = jax.lax.stop_gradient(potassium_peaks > min_req_value_mV)
    diffusion_peaks = jnp.max(diffusion_ei, axis=0)
    diffusion_elec_valid_bool = jax.lax.stop_gradient(diffusion_peaks > min_req_value_mV)

    sodium_betas = jnp.where(sodium_elec_valid_bool, peak_value / jnp.min(sodium_ei, axis=0), 0.0)
    potassium_betas = jnp.where(potassium_elec_valid_bool, peak_value / jnp.max(potassium_ei, axis=0), 0.0)
    diffusion_betas = jnp.where(diffusion_elec_valid_bool, peak_value / jnp.max(diffusion_ei, axis=0), 0.0)

    normalized_sodium_ei = jax.nn.softmax(sodium_betas[None, :] * sodium_ei, axis=0)
    normalized_potassium_ei = jax.nn.softmax(potassium_betas[None, :] * potassium_ei, axis=0)
    normalized_diffusion_ei = jax.nn.softmax(diffusion_betas[None, :] * diffusion_ei, axis=0)

    if component_one == 'sodium':
        normalized_ei_one = normalized_sodium_ei
        elec_valid_bool_one = sodium_elec_valid_bool
    elif component_one == 'diffusion':
        normalized_ei_one = normalized_diffusion_ei
        elec_valid_bool_one = diffusion_elec_valid_bool
    elif component_one == 'potassium':
        normalized_ei_one = normalized_potassium_ei
        elec_valid_bool_one = potassium_elec_valid_bool

    if component_two == 'sodium':
        normalized_ei_two = normalized_sodium_ei
        elec_valid_bool_two = sodium_elec_valid_bool
    elif component_two == 'diffusion':
        normalized_ei_two = normalized_diffusion_ei
        elec_valid_bool_two = diffusion_elec_valid_bool
    elif component_two == 'potassium':
        normalized_ei_two = normalized_potassium_ei
        elec_valid_bool_two = potassium_elec_valid_bool

    # Compute soft argmax (expected time index) for each component
    peak_times_one = soft_argmax(normalized_ei_one, axis=0)  # shape: (n_electrodes,)
    peak_times_two = soft_argmax(normalized_ei_two, axis=0)  # shape: (n_electrodes,)
    
    # Width is the time difference between the two components
    time_differences = peak_times_two - peak_times_one
    
    # If override electrodes are provided, use them directly (bypass amplitude checks)
    if precomputed_valid_elecs is not None:
        num_electrodes = ei.shape[1]
        precomputed_mask = jnp.zeros(num_electrodes, dtype=jnp.bool_)
        precomputed_mask = precomputed_mask.at[precomputed_valid_elecs].set(True)
        # Override: use only the precomputed electrodes, ignore amplitude checks
        valid_elec_bool = precomputed_mask
    else:
        # Default: Valid electrodes must have both components valid (based on amplitude)
        valid_elec_bool = elec_valid_bool_one & elec_valid_bool_two
    
    # Zero out invalid electrodes
    time_differences = jnp.where(valid_elec_bool, time_differences, 0.0)

    return time_differences, valid_elec_bool


def aligned_mse_loss(ei1: jnp.ndarray, ei2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the minimum mean squared error between two EIs over all possible time alignments.
    
    Args:
        ei1: First EI array of shape (time1, electrodes)
        ei2: Second EI array of shape (time2, electrodes)
        
    Returns:
        Minimum MSE over all alignments
    """
    # Ensure both EIs have the same number of electrodes
    assert ei1.shape[1] == ei2.shape[1], "EIs must have the same number of electrodes"
    
    num_electrodes = ei1.shape[1]
    num_offsets = ei1.shape[0] + ei2.shape[0] - 1
    
    # Pad ei1 along the time dimension to allow for all possible alignments
    # Pad with zeros on both sides
    pad_size = ei2.shape[0] - 1
    padded_ei1 = jnp.pad(ei1, ((pad_size, pad_size), (0, 0)), mode='constant', constant_values=0.0)
    
    # Compute MSE for each possible offset
    mse_values = jnp.zeros(num_offsets)
    
    for offset in range(num_offsets):
        # Extract the aligned portion of ei1
        aligned_ei1 = padded_ei1[offset:offset + ei2.shape[0], :]
        # Compute MSE for this alignment
        mse = jnp.mean(jnp.square(aligned_ei1 - ei2))
        mse_values = mse_values.at[offset].set(mse)
    
    # Return the minimum MSE
    return jnp.min(mse_values), mse_values
    


# def ei_widths(ei, max_time_offset:int) -> jnp.ndarray:
#     """
#     Compute the width of the EI at each electrode.
#     """
#     min_indices = jnp.argmin(ei, axis=0)
#     time_indices = jnp.arange(ei.shape[0])[:, None]
#     too_late_mask = time_indices - max_time_offset > min_indices
#     too_early_mask = time_indices + max_time_offset < min_indices

#     _first_masked_ei = jnp.where(too_late_mask, 0.0, ei)
#     masked_ei = jnp.where(too_early_mask, 0.0, _first_masked_ei)

#     squared_time_offsets_from_min = ((time_indices - min_indices)**2).astype(jnp.float32)

#     normalized_ei = jnp.abs(masked_ei) / jnp.sum(jnp.abs(masked_ei))

#     return jnp.sqrt(jnp.sum(squared_time_offsets_from_min * normalized_ei, axis=0))
