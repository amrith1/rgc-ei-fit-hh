"""
Visualization functions for stimulation response analysis.

This module provides functions to create 3D visualizations and planar cross sections
of multi-electrode stimulation response data.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_stimulation_response(amps, probs_data, title="3D Stimulation Response", 
                                figsize=(12, 8), electrode_labels=None, given_ax=None):
    """
    Create a 3D visualization of stimulation response probabilities.
    
    Parameters:
    -----------
    amps_gsort : np.ndarray
        Array of shape (n_points, 3) containing stimulation amplitudes for 3 electrodes
    probs_data : np.ndarray
        Array of shape (n_points,) containing response probabilities
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure (if None, figure is displayed)
    electrode_labels : list, optional
        List of 3 strings for electrode labels (default: ["Electrode 1", "Electrode 2", "Electrode 3"])
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """

    assert amps.shape[0] == probs_data.shape[0], "amps and probs_data must have the same number of rows"
    assert amps.shape[1] == 3, "amps must have 3 columns"

    if electrode_labels is None:
        electrode_labels = ["Electrode 1", "Electrode 2", "Electrode 3"]
    
    if given_ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = given_ax
        ax.set_3d_properties()
    
    scatter = ax.scatter(amps[:, 0], 
                        amps[:, 1],
                        amps[:, 2], 
                        c=probs_data,
                        cmap='viridis',
                        s=50,
                        alpha=0.6)
    
    ax.set_xlabel(f'{electrode_labels[0]} Amplitude (μA)')
    ax.set_ylabel(f'{electrode_labels[1]} Amplitude (μA)') 
    ax.set_zlabel(f'{electrode_labels[2]} Amplitude (μA)')
    
    plt.colorbar(scatter, label='Response Probability')
    plt.title(title)
    
    if given_ax is None:
        return fig, ax


def plot_stimulation_cross_sections(amps, probs_data, margin=0.2, 
                                   title="Stimulation Response Cross Sections",
                                   figsize=(20, 6), electrode_labels=None, given_axes=None):
    """
    Create planar cross sections of 3D stimulation response data.
    
    Parameters:
    -----------
    amps_gsort : np.ndarray
        Array of shape (n_points, 3) containing stimulation amplitudes for 3 electrodes
    probs_data : np.ndarray
        Array of shape (n_points,) containing response probabilities
    margin : float
        Points within +/- margin of 0 will be included in cross sections
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure (if None, figure is displayed)
    electrode_labels : list, optional
        List of 3 strings for electrode labels (default: ["Electrode 1", "Electrode 2", "Electrode 3"])
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """

    assert amps.shape[0] == probs_data.shape[0], "amps and probs_data must have the same number of rows"
    assert amps.shape[1] == 3, "amps must have 3 columns"

    if electrode_labels is None:
        electrode_labels = ["Electrode 1", "Electrode 2", "Electrode 3"]
    
    if given_axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        ax1, ax2, ax3 = axes
    else: #use the given axes
        ax1, ax2, ax3 = given_axes
    
    # Cross section restricting amplitude 1 (Electrode 1 ≈ 0)
    mask1 = (np.abs(amps[:, 0]) < margin) & (amps[:, 0] > 0)
    if np.sum(mask1) > 0:
        scatter1 = ax1.scatter(amps[mask1, 1], amps[mask1, 2], 
                              c=probs_data[mask1], cmap='viridis', alpha=0.6, s=30)
        ax1.set_xlabel(f'{electrode_labels[1]} Amplitude (μA)')
        ax1.set_ylabel(f'{electrode_labels[2]} Amplitude (μA)') 
        ax1.set_title(f'Cross section at {electrode_labels[0]} ≈ 0\n({np.sum(mask1)} points)')
        plt.colorbar(scatter1, ax=ax1, label='Response Probability')
    else:
        ax1.text(0.5, 0.5, 'No data points\nin this cross section', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Cross section at {electrode_labels[0]} ≈ 0')
    
    # Cross section restricting amplitude 2 (Electrode 2 ≈ 0)
    mask2 = (np.abs(amps[:, 1]) < margin) & (amps[:, 1] > 0)
    if np.sum(mask2) > 0:
        scatter2 = ax2.scatter(amps[mask2, 0], amps[mask2, 2],
                              c=probs_data[mask2], cmap='viridis', alpha=0.6, s=30)
        ax2.set_xlabel(f'{electrode_labels[0]} Amplitude (μA)')
        ax2.set_ylabel(f'{electrode_labels[2]} Amplitude (μA)')
        ax2.set_title(f'Cross section at {electrode_labels[1]} ≈ 0\n({np.sum(mask2)} points)')
        plt.colorbar(scatter2, ax=ax2, label='Response Probability')
    else:
        ax2.text(0.5, 0.5, 'No data points\nin this cross section', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Cross section at {electrode_labels[1]} ≈ 0')
    
    # Cross section restricting amplitude 3 (Electrode 3 ≈ 0)
    mask3 = (np.abs(amps[:, 2]) < margin) & (amps[:, 2] > 0)
    if np.sum(mask3) > 0:
        scatter3 = ax3.scatter(amps[mask3, 0], amps[mask3, 1],
                              c=probs_data[mask3], cmap='viridis', alpha=0.6, s=30)
        ax3.set_xlabel(f'{electrode_labels[0]} Amplitude (μA)') 
        ax3.set_ylabel(f'{electrode_labels[1]} Amplitude (μA)')
        ax3.set_title(f'Cross section at {electrode_labels[2]} ≈ 0\n({np.sum(mask3)} points)')
        plt.colorbar(scatter3, ax=ax3, label='Response Probability')
    else:
        ax3.text(0.5, 0.5, 'No data points\nin this cross section', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'Cross section at {electrode_labels[2]} ≈ 0')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if not given_axes:
        return fig, axes
    
def plot_2d_stimulation_response(amps_2d, probs_2d, elec_labels=None, 
                                figsize=(10, 8), cmap='viridis', 
                                title='2D Stimulation Response', 
                                xlabel='Current 1 (μA)', ylabel='Current 2 (μA)',
                                given_ax=None):
    """
    Plot a 2D scatter plot showing stimulation response for 2D current combinations.
    
    Args:
        amps_2d: 2D array of shape (n_amps, 2) containing current combinations
        probs_2d: 1D array of shape (n_amps) containing spike probabilities
        elec_labels: Optional list of electrode labels for the two dimensions
        figsize: Figure size tuple
        cmap: Colormap for the probability plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Assert that inputs have the correct shapes
    assert amps_2d.ndim == 2, f"Expected 2D amps_2d, got {amps_2d.ndim}D"
    assert probs_2d.ndim == 1, f"Expected 1D probs_2d, got {probs_2d.ndim}D"
    assert amps_2d.shape[1] == 2, f"Expected amps_2d to have 2 columns, got {amps_2d.shape[1]}"
    assert amps_2d.shape[0] == probs_2d.shape[0], f"Number of amplitude combinations ({amps_2d.shape[0]}) must match number of probabilities ({probs_2d.shape[0]})"
    
    # Extract the current values for each dimension
    amps1 = amps_2d[:, 0]  # First current dimension
    amps2 = amps_2d[:, 1]  # Second current dimension
    
    # Create the figure
    if given_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = given_ax
    
    # Create the scatter plot with colors based on probabilities
    scatter = ax.scatter(amps1, amps2, c=probs_2d, cmap=cmap, s=50, alpha=0.8, vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Spike Probability', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add electrode labels if provided
    if elec_labels is not None and len(elec_labels) >= 2:
        ax.text(0.02, 0.98, f'Electrodes: {elec_labels[0]}, {elec_labels[1]}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add zero lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Set axis limits to be symmetric around zero
    max_amp = max(np.abs(amps1).max(), np.abs(amps2).max())
    ax.set_xlim(-max_amp * 1.1, max_amp * 1.1)
    ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)
    
    plt.tight_layout()
    
    if given_ax is None:
        return fig, ax
