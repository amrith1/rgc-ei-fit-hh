from typing import List, Dict
import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from rgc_models.lib import HH, Ca, CaPump, KCa, CaNernstReversal, \
    get_baseline_triphasic_stimulus, point_source_stim_potential_mV_per_uA, \
    sigmoid, inverse_sigmoid, \
    get_filter_downsample_matrix

from rgc_models.loss_functions import sodium_peaks, diffusion_peaks, potassium_peaks,\
    global_curvature_loss, second_order_continuity_loss,\
    pairwise_time_differences, ei_widths, \
    precompute_valid_width_elecs, precompute_valid_pt_elecs, first_order_continuity_loss

from rgc_models.rgc_location_initialization import get_evenly_spaced_point_indices

from splinex.bspline import BSpline

jax.config.update('jax_platform_name', 'cpu')  # Force CPU usage
jax.config.update('jax_enable_x64', True)

def generate_assignment_vec(comp_indices_dict, region_values_dict, num_compartments:int):

    #assert that when you merge all the comp indices together, you get all indices from 0 to num_compartments - 1
    all_indices = jnp.concatenate(
        [indices for indices in comp_indices_dict.values() if indices.size > 0]
        )
    #assert all_indices.shape[0] == num_compartments and jnp.unique(all_indices).shape[0] == num_compartments, \
    #    "Number of compartments in comp_indices_dict does not match num_compartments"
    #generate the assignment vector
    assignment_vec = jnp.zeros(num_compartments)
    for region, comp_indices in comp_indices_dict.items():
        # Skip empty arrays
        if comp_indices.size > 0:
            assignment_vec = assignment_vec.at[comp_indices].set(region_values_dict[region])
    return assignment_vec

class RGC:
    def __init__(self, electrode_locations_um:jnp.ndarray, num_control_points:int,
                 TIME_STEP=10e-3, TOTAL_SIM_TIME=1700e-3, ESTIM_TIME_STEP=5e-3, \
                 total_axon_length=520.0, axon_comp_length=4.0, control_point_spacing=40.0,
                 important_elecs:List[int]=None, soma_voltage_clamp:float=-50.0,
                 elec_pairwise_dist_threshold:float=35.0, #um, timing diffs wont be considered for electrodes further than this
                 timing_features_min_amp_mV:float=0.01, #roughly 2 ADC counts
                 ):
        self.TIME_STEP = TIME_STEP
        self.SIM_TIME_SAMPLES = int(TOTAL_SIM_TIME / TIME_STEP)
        self.ESTIM_TIME_STEP = ESTIM_TIME_STEP
        self.MEM_CAPACITANCE = 1.0 #uF/cm^2
        self.axon_comp_length = axon_comp_length
        self.E_PAS = -65.02
        self.LITKE_TIME_STEP = 50e-3
        self.ELECTRODE_LOCATIONS_UM = electrode_locations_um
        
        self.stim_sample_cutoff = int(150e-3 / TIME_STEP)

        self.filter_matrix = get_filter_downsample_matrix(
            num_inputs=int(TOTAL_SIM_TIME / TIME_STEP) - self.stim_sample_cutoff,
            num_taps=101,
            fs=1e3/self.TIME_STEP,
            low_cut_freq=4e1,
            high_cut_freq=4e3,
            downsample_factor=int(1)
        ).transpose()

        self.cell_params_list = ["HH_gNa", "HH_gK", "radius", "length"]

        self.BASELINE_LENGTHS = {
            'soma': 20.0,
            'AH': 40.0,
            'SOCB': 40.0,
            'NR': 80.0,
        }
        self.BASELINE_LENGTHS['axon'] = total_axon_length - self.BASELINE_LENGTHS['AH']\
              - self.BASELINE_LENGTHS['SOCB'] - self.BASELINE_LENGTHS['NR']

        assert self.BASELINE_LENGTHS['axon'] > 0.0
        assert total_axon_length % axon_comp_length == 0

        self.BASELINE_DEPTHS = {
            'soma': -30.0,
            'axon': -10.0
        }

        self.axial_resistivity_ohm_cm = jnp.array([143.0])

        self.ei_cell, self.adjacency_matrix, self.section_indices, \
            self.axon_indices_division, self.non_axon_relative_locs, self.full_axon_notches, self.non_axon_lengths = \
            self.build_cell(self.BASELINE_LENGTHS, self.BASELINE_DEPTHS, self.axon_comp_length)
        
        self.estim_cell, _, _, _, _, _, _ = self.build_cell(self.BASELINE_LENGTHS, self.BASELINE_DEPTHS, self.axon_comp_length)

        ei_initial_stimulus_voltage = jnp.zeros(len(self.ei_cell.nodes)) + self.E_PAS
        #ei_initial_stimulus_voltage = ei_initial_stimulus_voltage.at[self.section_indices['soma']].set(-50.0)
        soma_voltages = jnp.linspace(soma_voltage_clamp, soma_voltage_clamp, len(self.section_indices['soma']))
        ei_initial_stimulus_voltage = ei_initial_stimulus_voltage.at[self.section_indices['soma']].set(soma_voltages)
        
        self.ei_cell.set("v", ei_initial_stimulus_voltage)
        self.ei_cell.stimulate(jnp.zeros((len(self.ei_cell.nodes), self.SIM_TIME_SAMPLES)))
        self.ei_cell_record_values = ["v", "i_HH", "i_Ca", "i_KCa"]
        for record_value in self.ei_cell_record_values:
            self.ei_cell.record(record_value)
        
        self.estim_cell_record_values = ["v"]
        for record_value in self.estim_cell_record_values:
            self.estim_cell.record(record_value)

        PRE_TIME = 50e-3 #ms
        END_ACTUAL_STIM_TIME = PRE_TIME + 150e-3 #ms
        INDICATOR_TIME = END_ACTUAL_STIM_TIME + 20e-3 #ms
        self.estim_indicator_sample = int(INDICATOR_TIME / self.ESTIM_TIME_STEP)
        self.triphasic_stimulus = get_baseline_triphasic_stimulus(
            t_pre=PRE_TIME, t_max=1500e-3, time_step=self.ESTIM_TIME_STEP)
        
        self.num_total_compartments = len(self.ei_cell.nodes)
        self.electrode_locations_um = electrode_locations_um

        self.num_axon_compartments = len(self.axon_indices_division['full_axon'])
        self.total_axon_length = self.axon_comp_length * self.num_axon_compartments
        self.control_point_spacing = control_point_spacing #um

        self.num_control_points = num_control_points
        INTERP_US_FACTOR = int(10)
        self.spline_interpolator = BSpline(n_in=self.num_control_points, 
                n_out=self.num_control_points * INTERP_US_FACTOR * int(self.control_point_spacing/self.axon_comp_length), 
                degree=3)

        # Get axon notch locations by interpolating the control points
        self.soma_smooth_dist_um = 200.0
        self.even_axon_spline_index = int(jnp.ceil(self.soma_smooth_dist_um / self.control_point_spacing))

        self.ei_feature_weights = {
            'sodium_peaks': 5e5,
            'diffusion_peaks':2e5,
            'potassium_peaks':1e5,
            'sodium_velocity': 1e-1 * ((self.TIME_STEP/10e-3) ** 2),
            'ei_widths_so_po': 3e-8 * ((self.TIME_STEP/10e-3) ** 2),
            'ei_widths_di_so': 3e-8 * ((self.TIME_STEP/10e-3) ** 2),
        }

        self.penalty_weights = {
            'first_order_continuity': 1e0 * (80.0/self.control_point_spacing) ** 2,
            'second_order_continuity': 5e-2 * (80.0/self.control_point_spacing) ** 2,
            'axon_rise': 2e-1 * (80.0/self.control_point_spacing) ** 2,
        }

        self.estim_loss_weight = 100.0

        _num_elecs = self.electrode_locations_um.shape[0]
        self.loss_elec_weightings = jnp.ones(_num_elecs, dtype=jnp.float32)
        self.loss_elec_adjacency_weightings = jnp.ones((_num_elecs, _num_elecs), dtype=jnp.float32)
        self.elec_pairwise_distances = jnp.linalg.norm(
            self.electrode_locations_um[None, :, :] - self.electrode_locations_um[:, None, :], axis=2
            )
        self.elec_proximity_mask = self.elec_pairwise_distances < elec_pairwise_dist_threshold
        print(f'Number of proximal electrode pairs: {jnp.sum(self.elec_proximity_mask)}', flush=True)
        self.velocity_elecs_override_used = False  # Track if velocity override is being used
        # When set via set_training_loss_features, loss_v1/v2 skip ablated ops (not just weight=0).
        self._training_ei_features = None  # None => all EI features traced
        self._training_estim_enabled = True
        
        IMPORTANT_ELEC_WEIGHT = 10.0
        if important_elecs is not None:
            _elecs = jnp.array(important_elecs, dtype=jnp.int32)
            self.loss_elec_weightings = \
                self.loss_elec_weightings.at[_elecs].set(IMPORTANT_ELEC_WEIGHT)
            for _e1 in _elecs:
                for _e2 in _elecs:
                    self.loss_elec_adjacency_weightings = \
                        self.loss_elec_adjacency_weightings.at[_e1, _e2].set(IMPORTANT_ELEC_WEIGHT)

        self.gt_features = None
        self.loss_precomputes = None
        self.timing_features_min_amp_mV = timing_features_min_amp_mV

        self.jited_predict_ei = jax.jit(self.predict_ei)
        self.jitted_loss = jax.jit(self.loss_v1)
        self.jitted_grad = jax.jit(jax.value_and_grad(self.loss_v1, argnums=0, has_aux=True))
        self.jitted_grad_full = jax.jit(jax.value_and_grad(self.loss_full, argnums=0, has_aux=True))
        
        # GradNorm-compatible loss with weights as arguments
        self.jitted_loss_v2 = jax.jit(self.loss_v2)
        self.jitted_grad_v2 = jax.jit(jax.value_and_grad(self.loss_v2, argnums=0, has_aux=True))
        self.jitted_single_elec_stim = jax.jit(self.triphasic_xtra_singleelec_stim)
        self.jitted_reduced_single_elec_stim = jax.jit(self.reduced_triphasic_xtra_singleelec_stim)
        self.jitted_binary_search_threshold = jax.jit(self.binary_search_threshold)
        self.jitted_binary_search_threshold_fast = jax.jit(self.binary_search_threshold_fast)

        self.jitted_estim_single_diff_prob = jax.jit(self.estim_single_elec_differentiable_probability)
        self.jitted_estim_single_diff_prob_vmapped = jax.vmap(self.jitted_estim_single_diff_prob, in_axes=(None, 0, 0, 0))

        self.jitted_multielec_inference = jax.jit(self.is_spike_triphasic_xtra_multielec_stim)
        self.jitted_multielec_inference_vmapped = jax.vmap(self.jitted_multielec_inference, in_axes=(None, None, 0))

        self.jitted_estim_loss = jax.jit(self.estim_loss)
        self.jitted_estim_loss_grad = jax.jit(jax.value_and_grad(self.estim_loss, argnums=0, has_aux=False))

    def build_cell(self, baseline_lengths, baseline_depths, axon_comp_length):
        """
        Build an RGC, currently just two branches,
        one for soma, one for axon.
        """
        #first lets make the soma branch
        SOMA_COMP_LENGTH = 2.0 #um
        num_soma_compartments = int(baseline_lengths['soma']/SOMA_COMP_LENGTH)
        total_axon_length = baseline_lengths['AH'] + baseline_lengths['SOCB'] \
            + baseline_lengths['NR'] + baseline_lengths['axon']
        assert total_axon_length % axon_comp_length == 0, \
            f"Total axon length must be divisible by axon compartment length, total_axon_length: {total_axon_length}, axon_comp_length: {axon_comp_length}"
        num_axon_compartments = int(total_axon_length/axon_comp_length)
        num_compartments = {
            'soma': int(num_soma_compartments),
            'AH': int(baseline_lengths['AH']/axon_comp_length),
            'SOCB': int(baseline_lengths['SOCB']/axon_comp_length),
            'NR': int(baseline_lengths['NR']/axon_comp_length),
            'axon': int(baseline_lengths['axon']/axon_comp_length)
        }

        cell = jx.Cell([
            jx.Branch([jx.Compartment()]*num_compartments['soma']),
            jx.Branch([jx.Compartment()]*num_compartments['AH']),
            jx.Branch([jx.Compartment()]*num_compartments['SOCB']),
            jx.Branch([jx.Compartment()]*num_compartments['NR']),
            jx.Branch([jx.Compartment()]*num_compartments['axon'])
        ], parents=[-1, 0, 1, 2, 3])

        section_indices = {
            'soma': jnp.arange(num_soma_compartments, dtype=jnp.int32),
            'AH': jnp.arange(num_soma_compartments, num_soma_compartments + num_compartments['AH'], dtype=jnp.int32),
            'SOCB': jnp.arange(num_soma_compartments + num_compartments['AH'], \
                num_soma_compartments + num_compartments['AH'] + num_compartments['SOCB'], dtype=jnp.int32),
            'NR': jnp.arange(num_soma_compartments + num_compartments['AH'] + num_compartments['SOCB'],\
                 num_soma_compartments + num_compartments['AH'] + num_compartments['SOCB'] + num_compartments['NR'], dtype=jnp.int32),
            'axon': jnp.arange(num_soma_compartments + num_compartments['AH'] + num_compartments['SOCB'] + num_compartments['NR'], \
                num_soma_compartments + num_compartments['AH'] + num_compartments['SOCB'] + num_compartments['NR'] + num_compartments['axon'], dtype=jnp.int32),
            'dendrite': jnp.array([], dtype=jnp.int32)
        }
        non_axon_indices = section_indices['soma']

        cell.insert(HH())
        cell.insert(Ca())
        cell.insert(CaPump())
        cell.insert(KCa())
        cell.insert(CaNernstReversal())

        for param, cell_wide_val in [('HH_gLeak', 1e-4), ('HH_eNa', 60.60), ('HH_eK', -101.34),\
                                    ('HH_eLeak', -65.02), ('axial_resistivity', self.axial_resistivity_ohm_cm[0]), ('v', self.E_PAS),\
                                    ("HH_m", 0.0353), ("HH_h", 0.9054), ("HH_n", 0.0677), ("capacitance", self.MEM_CAPACITANCE), ("Ca_c", 0.0019)]:
            cell.set(param, cell_wide_val)

        cell.set("gCa", generate_assignment_vec(section_indices,
            {'dendrite': 0.001, 'soma': 0.00075, 'AH': 0.00075, 'SOCB': 0.00075, 'NR': 0.00075, 'axon': 0.00075},
            len(cell.nodes)))
        
        cell.set("gKCa", generate_assignment_vec(section_indices,
            {'dendrite': 0.00017, 'soma': 0.00017, 'AH': 0.00017, 'SOCB': 0.00011, 'NR': 0.0002, 'axon': 0.0002},
            len(cell.nodes)))

        for param in self.cell_params_list:
            cell.comp("all").make_trainable(param)

        defaults = self.get_default_kish_2023_values_dict()
        
        cell.set("length", generate_assignment_vec(section_indices, 
            {'dendrite': SOMA_COMP_LENGTH, 'soma': SOMA_COMP_LENGTH, 'AH': axon_comp_length, 'SOCB': axon_comp_length, 'NR': axon_comp_length, 'axon': axon_comp_length}, 
            len(cell.nodes)))
        
        cell.set("HH_gNa", generate_assignment_vec(section_indices, 
            {'dendrite': defaults['HH_gNa_dendrite'], 'soma': defaults['HH_gNa_soma'],
              'AH': defaults['HH_gNa_AH'], 'SOCB': defaults['HH_gNa_SOCB'], 
              'NR': defaults['HH_gNa_NR'], 'axon': defaults['HH_gNa_axon']}, 
            len(cell.nodes)))
        
        cell.set("HH_gK", generate_assignment_vec(section_indices, 
            {'dendrite': defaults['HH_gK_dendrite'], 'soma': defaults['HH_gK_soma'],
              'AH': defaults['HH_gK_AH'], 'SOCB': defaults['HH_gK_SOCB'], 
              'NR': defaults['HH_gK_NR'], 'axon': defaults['HH_gK_axon']}, 
            len(cell.nodes)))
        
        # Create linear taper from AH radius (1.5) to NR radius (0.4) for SOCB section
        socb_radii = jnp.linspace(defaults['radius_AH'], defaults['radius_NR'], len(section_indices['SOCB']))
        #print(socb_radii)
        cell.set("radius", generate_assignment_vec(section_indices, 
            {'dendrite': defaults['radius_dendrite'], 'soma': defaults['radius_soma'],
              'AH': defaults['radius_AH'], 'SOCB': socb_radii, 'NR': defaults['radius_NR'], 
              'axon': defaults['radius_axon']}, 
            len(cell.nodes)))

        total_ais_compartments = len(section_indices['AH']) + len(section_indices['SOCB'])
        AIS_RUN = np.sqrt(80.0**2 - (baseline_depths['soma'] - baseline_depths['axon'])**2)
        AIS_START = [0.0, 0.0, baseline_depths['soma']]
        AIS_END = [AIS_RUN, 0.0, baseline_depths['axon']]

        # Generate intermediate AIS locations including both endpoints
        ais_notches = jnp.stack([
            jnp.linspace(AIS_START[0], AIS_END[0], total_ais_compartments + 1),
            jnp.linspace(AIS_START[1], AIS_END[1], total_ais_compartments + 1), 
            jnp.linspace(AIS_START[2], AIS_END[2], total_ais_compartments + 1)
        ], axis=1)

        total_axon_compartments = len(section_indices['axon']) + len(section_indices['NR'])
        axon_notches = jnp.stack([
            jnp.linspace(AIS_END[0], AIS_END[0] + total_axon_compartments * axon_comp_length, total_axon_compartments + 1),
            jnp.linspace(AIS_END[1], AIS_END[1], total_axon_compartments + 1), 
            jnp.linspace(AIS_END[2], AIS_END[2], total_axon_compartments + 1)
        ], axis=1)

        full_axon_notches = jnp.concatenate([ais_notches, axon_notches[1:]])

        #defined relative to the first axon notch        
        HALF_SOMA_COMP = SOMA_COMP_LENGTH/2.0

        soma_relative_locs = jnp.stack([
            jnp.linspace(0.0, 0.0, num_soma_compartments),
            jnp.linspace(0.0, 0.0, num_soma_compartments),
            jnp.linspace(-HALF_SOMA_COMP - (num_soma_compartments - 1) * SOMA_COMP_LENGTH, 
                         -HALF_SOMA_COMP,
                         num_soma_compartments)
        ], axis=1)

        full_axon_indices = jnp.concatenate([section_indices['AH'], section_indices['SOCB'], section_indices['NR'], section_indices['axon']])
        non_axon_indices = jnp.concatenate([section_indices['soma'], section_indices['dendrite']])

        non_axon_relative_locs = soma_relative_locs
        axon_indices_division = {
            'full_axon': full_axon_indices,
            'non_axon': non_axon_indices,
        }

        non_axon_lengths = jnp.array(cell.nodes['length'][non_axon_indices])

        adjacency_matrix = jnp.zeros((len(cell.nodes), len(cell.nodes))).astype(jnp.bool_)
        # Set adjacent compartments to 1 in adjacency matrix
        for i in range(len(cell.nodes)-1):
            adjacency_matrix = adjacency_matrix.at[i,i+1].set(True)
            adjacency_matrix = adjacency_matrix.at[i+1,i].set(True)

        return cell, adjacency_matrix, section_indices, \
            axon_indices_division, non_axon_relative_locs, full_axon_notches, non_axon_lengths
    
    def predict_ei(self, full_params, seed = 0):
        """
        Predict extracellular potentials using the current parameters.
        """
        cell_params = [{param_name: full_params[param_name]} for param_name in self.cell_params_list]

        # Handle noise level
        noise_level = full_params.get("noise_level", 0.0)

        # PRNG key
        key = jax.random.PRNGKey(seed)

        # Generate noisy input current
        # Shape should be (num_compartments, time_samples) for jaxley data_stimulate
        noisy_current = noise_level * jax.random.normal(
            key,
            shape=(self.num_total_compartments, self.SIM_TIME_SAMPLES),
        )

        data_stimuli = self.ei_cell.data_stimulate(noisy_current)

        outputs = jx.integrate(self.ei_cell, delta_t=self.TIME_STEP, 
                    params=cell_params, #voltage_solver="jaxley.stone"
                    data_stimuli = data_stimuli,
                    ).reshape(
            len(self.ei_cell_record_values), self.num_total_compartments, -1)
        
        _v, _i_HH = outputs[0, :, :], outputs[1, :, :]
        _i_Ca, _i_KCa = outputs[2, :, :], outputs[3, :, :]

        capacitive_current = self.MEM_CAPACITANCE * jnp.diff(_v, axis=1) / (self.TIME_STEP * 1e3) #convert to mA/cm^2
        mem_current_per_area = _i_HH[:, 0:-1] + capacitive_current + _i_Ca[:, 0:-1] + _i_KCa[:, 0:-1] #mA/cm^2
        mem_voltage = _v[:, 0:-1]
        
        compartment_surface_area = jax.device_put(2 * jnp.pi * full_params["radius"] * full_params["length"]) #um^2
        mem_current_uA = 1e-5 * mem_current_per_area * compartment_surface_area[:, None]# still (num_compartments, num_timepoints)

        current_to_eap_matrix = point_source_stim_potential_mV_per_uA(
            extra_resistivity_ohm_cm=full_params['extra_resistivity_ohm_cm'], 
            source_locations_um=full_params["compartment_locations"], 
            record_locations_um=self.electrode_locations_um
            ) #shape: (n_electrodes, n_compartments)

        eap_mV_pre_filter = current_to_eap_matrix @ mem_current_uA
        eap_mV = eap_mV_pre_filter[:, self.stim_sample_cutoff:] @ self.filter_matrix

        #time should be the first dimension
        return eap_mV.transpose(), mem_current_per_area.transpose(), \
            mem_voltage.transpose(), eap_mV_pre_filter.transpose()

    def inverse_sigmoid_params_v1(self, params):
        param_bounds = self.get_param_bounds_v1()
        depth_bounds = self.get_depth_bounds_v1()

        opt_params = {}
        for param_name, (lower, upper) in param_bounds.items():
            assert lower < upper, f"Lower bound {lower} is not less than upper bound {upper} for {param_name}"
            assert params[param_name] > lower and params[param_name] < upper, f"Parameter {param_name} is not within bounds {lower} and {upper}"
            opt_params[param_name] = inverse_sigmoid(params[param_name], lower, upper)
        #axon_spline_control_points = params['axon_spline_control_points']

        opt_depth = inverse_sigmoid(params['axon_spline_control_points_z'], depth_bounds[0], depth_bounds[1])
        opt_params['axon_spline_control_points_z'] = opt_depth
        opt_params['axon_spline_control_points_xy'] = params['axon_spline_control_points_xy']

        return opt_params

    def forward_sigmoid_params_v1(self, opt_params):
        param_bounds = self.get_param_bounds_v1()
        depth_bounds = self.get_depth_bounds_v1()
    
        params = {}
        for param_name, (lower, upper) in param_bounds.items():
            assert lower < upper, f"Lower bound {lower} is not less than upper bound {upper} for {param_name}"
            params[param_name] = sigmoid(opt_params[param_name], lower, upper)

        opt_axon_spline_control_points_z = opt_params['axon_spline_control_points_z']
        depth = sigmoid(opt_axon_spline_control_points_z, depth_bounds[0], depth_bounds[1])

        #enforce that all the z coordinates for the axon are the same after this index by taking an average
        average_axon_depth = jnp.mean(depth[self.even_axon_spline_index:])
        
        smoothed_axon_z_control_points = jnp.concatenate(
            [depth[:self.even_axon_spline_index],
             jnp.ones(self.num_control_points - self.even_axon_spline_index) * average_axon_depth], axis=0)
        
        assert smoothed_axon_z_control_points.shape == depth.shape

        params['axon_spline_control_points_z'] = smoothed_axon_z_control_points

        og_x = opt_params['axon_spline_control_points_xy'][:, 0]
        og_y = opt_params['axon_spline_control_points_xy'][:, 1]

        smoothed_init_x = jnp.linspace(og_x[0], og_x[self.even_axon_spline_index - 1], self.even_axon_spline_index)
        smoothed_init_y = jnp.linspace(og_y[0], og_y[self.even_axon_spline_index - 1], self.even_axon_spline_index)

        params['axon_spline_control_points_xy'] = jnp.concatenate([
            jnp.stack([smoothed_init_x, smoothed_init_y], axis=1),
            opt_params['axon_spline_control_points_xy'][self.even_axon_spline_index:]
        ], axis=0)

        return params

    
    def expand_params_v1(self, params):
        """
        Currently, predict_ei takes in parameters as full vectors, but we want to have fewer trainable parameters.
        Here, input params is a dictionary of parameters, and we want to expand it to the format required by predict_ei.

        Entries of params are:
        - HH_gNa (for every section)
        - HH_gK (for every section)
        - radius (for every section)
        - a global radius scale factor
        - axon notch control points for spline interpolation

        We will use the generate_assignment_vec function to expand the parameters.
        """

        # First, expand the HH_gNa, HH_gK, radius
        full_params = {
            'axial_resistivity': self.axial_resistivity_ohm_cm,
            'positive_current_adjustment': 0.55 * jnp.ones(self.ELECTRODE_LOCATIONS_UM.shape[0]),
            'negative_current_adjustment': 0.55 * jnp.ones(self.ELECTRODE_LOCATIONS_UM.shape[0]),
            'extra_resistivity_ohm_cm': params['extra_resistivity_ohm_cm'],
        }
        full_params['HH_gNa'] = generate_assignment_vec(self.section_indices,
            {'dendrite': params['HH_gNa_dendrite'], 'soma': params['HH_gNa_soma'], 'AH': params['HH_gNa_AH'], 'SOCB': params['HH_gNa_SOCB'], 'NR': params['HH_gNa_NR'], 'axon': params['HH_gNa_axon']},
            len(self.ei_cell.nodes))
        full_params['HH_gK'] = generate_assignment_vec(self.section_indices,
            {'dendrite': params['HH_gK_dendrite'], 'soma': params['HH_gK_soma'], 'AH': params['HH_gK_AH'], 'SOCB': params['HH_gK_SOCB'], 'NR': params['HH_gK_NR'], 'axon': params['HH_gK_axon']},
            len(self.ei_cell.nodes))
        
        socb_radii = jnp.linspace(params['radius_AH'], params['radius_NR'], len(self.section_indices['SOCB']))
        unscaled_radii = generate_assignment_vec(self.section_indices,
            {'dendrite': params['radius_dendrite'], 'soma': params['radius_soma'], 'AH': params['radius_AH'], 'SOCB': socb_radii, 'NR': params['radius_NR'], 'axon': params['radius_axon']},
            len(self.ei_cell.nodes))
        
        full_params['radius'] = unscaled_radii * params['radius_scale_factor']

        axon_spline_control_points = jnp.concatenate(
            [params['axon_spline_control_points_xy'],
             params['axon_spline_control_points_z'][:, None]], axis=1
        )

        hd_axon_notch_locs, _, _ = self.spline_interpolator(axon_spline_control_points)

        notch_indices = get_evenly_spaced_point_indices(
            hd_axon_notch_locs, 
            spacing_um=self.axon_comp_length, 
            num_points=self.num_axon_compartments + 1)

        axon_notch_locs = hd_axon_notch_locs[notch_indices, :]

        #full_axon_lengths = jnp.linalg.norm(axon_notch_locs[1:, :] - axon_notch_locs[0:-1, :], axis=1)
        #holding this to be constant for now
        full_axon_lengths = jnp.ones(self.num_axon_compartments) * self.axon_comp_length
        full_params['length'] = generate_assignment_vec(self.axon_indices_division,
            {'full_axon': full_axon_lengths, 'non_axon': self.non_axon_lengths},
            self.num_total_compartments)

        # Now, expand the axon notch locations
        non_axon_locations = self.non_axon_relative_locs + axon_notch_locs[0]
        full_axon_locations = (axon_notch_locs[1:, :] +  axon_notch_locs[0:-1, :])/2.0
        
        # Now, expand the axon notch locations
        full_params['compartment_locations'] = jnp.vstack([
            generate_assignment_vec(self.axon_indices_division, \
                {'full_axon': full_axon_locations[:, 0], 'non_axon': non_axon_locations[:, 0]}, self.num_total_compartments),
            generate_assignment_vec(self.axon_indices_division, \
                {'full_axon': full_axon_locations[:, 1], 'non_axon': non_axon_locations[:, 1]}, self.num_total_compartments),
            generate_assignment_vec(self.axon_indices_division, \
                {'full_axon': full_axon_locations[:, 2], 'non_axon': non_axon_locations[:, 2]}, self.num_total_compartments)
        ]).T

        # Add noise_level
        if "noise_level" in params:
            full_params["noise_level"] = params["noise_level"]

        return full_params
    
    def _is_ei_feature_enabled(self, feature_name: str) -> bool:
        if self._training_ei_features is None:
            return True
        return feature_name in self._training_ei_features

    def set_training_loss_features(self, enabled_loss_features: frozenset[str]) -> None:
        """Control which EI feature ops and estim are traced in loss_v1/v2 during training."""
        self._training_ei_features = frozenset(
            name for name in self.ei_feature_weights if name in enabled_loss_features
        )
        self._training_estim_enabled = 'estim' in enabled_loss_features

    def get_ei_features(self, eap, *, training_only: bool = False):
        """
        Get the features of the EAP.

        training_only: when True, skip EI features disabled via set_training_loss_features
        (used in loss_v1/v2 so ablated terms are not differentiated).
        """
        features = {}
        for feature_name in self.ei_feature_weights.keys():
            if training_only and not self._is_ei_feature_enabled(feature_name):
                continue
            if feature_name == 'sodium_peaks':
                features[feature_name] = sodium_peaks(eap)
            elif feature_name == 'diffusion_peaks':
                features[feature_name] = diffusion_peaks(eap)
            elif feature_name == 'potassium_peaks':
                features[feature_name] = potassium_peaks(eap)
            elif feature_name == 'sodium_velocity':
                features[feature_name] = pairwise_time_differences(eap, 
                    component='sodium', min_req_value_mV=self.timing_features_min_amp_mV,
                    precomputed_valid_elec_i_indices=self.loss_precomputes['sodium_velocity_valid_elec_pairs'][0],
                    precomputed_valid_elec_j_indices=self.loss_precomputes['sodium_velocity_valid_elec_pairs'][1]
                    )
            elif feature_name == 'ei_widths_so_po':
                features[feature_name] = ei_widths(eap, 
                    component_one='sodium', component_two='potassium', 
                    min_req_value_mV=self.timing_features_min_amp_mV, 
                    precomputed_valid_elecs=self.loss_precomputes['ei_widths_so_po_valid_elecs']
                    )
            elif feature_name == 'ei_widths_di_so':
                features[feature_name] = ei_widths(eap, 
                    component_one='diffusion', component_two='sodium', 
                    min_req_value_mV=self.timing_features_min_amp_mV, 
                    precomputed_valid_elecs=self.loss_precomputes['ei_widths_di_so_valid_elecs']
                    )
            else:
                raise ValueError(f"Feature {feature_name} not found")
        return features
    
    def set_gt_ei_features(self, eap, velocity_elecs_override=None, 
                          ei_widths_so_po_elecs_override=None, 
                          ei_widths_di_so_elecs_override=None):
        # Compute default width electrodes
        ei_widths_so_po_valid_elecs = precompute_valid_width_elecs(
            eap, min_req_value_mV=self.timing_features_min_amp_mV, 
            component_one='sodium', component_two='potassium'
            )
        
        ei_widths_di_so_valid_elecs = precompute_valid_width_elecs(
            eap, min_req_value_mV=self.timing_features_min_amp_mV, 
            component_one='diffusion', component_two='sodium'
            )

        # Apply overrides if provided
        if ei_widths_so_po_elecs_override is not None:
            ei_widths_so_po_valid_elecs = jnp.array(ei_widths_so_po_elecs_override)
        
        if ei_widths_di_so_elecs_override is not None:
            # If empty array or None is passed, ignore this feature entirely
            if len(ei_widths_di_so_elecs_override) == 0:
                ei_widths_di_so_valid_elecs = jnp.array([], dtype=jnp.int32)
            else:
                ei_widths_di_so_valid_elecs = jnp.array(ei_widths_di_so_elecs_override)

        sodium_velocity_valid_elec_pairs = precompute_valid_pt_elecs(
            eap, min_req_value_mV=self.timing_features_min_amp_mV, 
            component='sodium', additional_mask=self.elec_proximity_mask
            )
        
        self.velocity_elecs_override_used = False
        if velocity_elecs_override is not None:
            sodium_velocity_valid_elec_pairs = velocity_elecs_override
            self.velocity_elecs_override_used = True

        self.loss_precomputes = {
            'ei_widths_so_po_valid_elecs': ei_widths_so_po_valid_elecs,
            'ei_widths_di_so_valid_elecs': ei_widths_di_so_valid_elecs,
            'sodium_velocity_valid_elec_pairs': sodium_velocity_valid_elec_pairs
        }
        print(f"number of ei_widths_so_po_valid_elecs: {ei_widths_so_po_valid_elecs.size}", flush=True)
        print(f"number of ei_widths_di_so_valid_elecs: {ei_widths_di_so_valid_elecs.size}", flush=True)
        print(f"number of sodium_velocity_valid_elec_pairs: {sodium_velocity_valid_elec_pairs[0].shape}, {sodium_velocity_valid_elec_pairs[1].shape}", flush=True)
        self.gt_features = self.get_ei_features(eap)
    
    def get_penalties(self, params):
        penalties = {}
        for penalty_name in self.penalty_weights.keys():
            if penalty_name == 'first_order_continuity':
                penalties[penalty_name] = \
                    first_order_continuity_loss(params['axon_spline_control_points_xy'],
                                                compartment_spacing=self.control_point_spacing)
            elif penalty_name == 'second_order_continuity':
                #only penalize the x and y coordinates
                penalties[penalty_name] = \
                    second_order_continuity_loss(params['axon_spline_control_points_xy'])
            elif penalty_name == 'axon_rise':
                # Penalize if z coordinates don't increase monotonically
                z_diffs = jnp.diff(params['axon_spline_control_points_z'])
                penalties[penalty_name] = jnp.mean(jnp.square(jnp.maximum(-z_diffs, 0.0)))
            else:
                raise ValueError(f"Penalty {penalty_name} not found")
        return penalties

    def estim_loss(self, full_params, estim_data):

        pred_probs = self.jitted_estim_single_diff_prob_vmapped(full_params, estim_data['elecs'], estim_data['intensities'], estim_data['critical_thresholds'])
        loss = jnp.mean(jnp.square(pred_probs - estim_data['gt_probs']))
        return loss

    def loss_v1(self, opt_params, estim_data):
        params = self.forward_sigmoid_params_v1(opt_params)
        full_params = self.expand_params_v1(params)
        eap, mem_current_per_area, mem_voltage, eap_pre_filter = self.jited_predict_ei(full_params)

        if self._training_estim_enabled:
            estim_loss = self.estim_loss_weight * self.estim_loss(full_params, estim_data)
        else:
            estim_loss = 0.0

        eap_features = self.get_ei_features(eap, training_only=True)
        penalties = self.get_penalties(params)
        feature_losses = {}

        for feature_name, weight in self.ei_feature_weights.items():
            if not self._is_ei_feature_enabled(feature_name):
                feature_losses[feature_name] = 0.0
                continue
            if feature_name in ['sodium_peaks', 'diffusion_peaks', 'potassium_peaks']:
                square_error = jnp.square(eap_features[feature_name] - self.gt_features[feature_name])
                weighted_square_error = weight * square_error * self.loss_elec_weightings
                feature_losses[feature_name] = jnp.mean(weighted_square_error)

            elif feature_name in ['ei_widths_so_po', 'ei_widths_di_so']:
                width_gt, valid_elec_bool_gt = self.gt_features[feature_name]
                width_pred, valid_elec_bool_pred = eap_features[feature_name]
                valid_elec_bool = valid_elec_bool_gt & valid_elec_bool_pred
                square_error = jnp.where(valid_elec_bool, jnp.square(width_pred - width_gt), 0.0)
                weighted_square_error = square_error * self.loss_elec_weightings * weight
                # Handle case where no valid electrodes (empty override) - loss is 0
                num_valid_gt = jnp.sum(valid_elec_bool_gt)
                feature_losses[feature_name] = jnp.where(num_valid_gt > 0,
                                                         jnp.sum(weighted_square_error) / num_valid_gt,
                                                         0.0)

            elif feature_name in ['sodium_velocity']:
                velocity_gt, _, valid_elec_pairs_bool_gt = self.gt_features[feature_name]
                velocity_pred, _, valid_elec_pairs_bool_pred = eap_features[feature_name]
                # Don't apply proximity mask if override was used (override pairs may be far apart)
                if self.velocity_elecs_override_used:
                    valid_elec_pairs_bool = valid_elec_pairs_bool_gt & valid_elec_pairs_bool_pred
                else:
                    valid_elec_pairs_bool = valid_elec_pairs_bool_gt & valid_elec_pairs_bool_pred & self.elec_proximity_mask
                square_error = jnp.where(valid_elec_pairs_bool, jnp.square(velocity_pred - velocity_gt), 0.0)
                weighted_square_error = square_error * self.loss_elec_adjacency_weightings * weight
                # Handle case where no valid pairs (division by zero protection)
                num_valid_gt = jnp.sum(valid_elec_pairs_bool_gt)
                feature_losses[feature_name] = jnp.where(num_valid_gt > 0,
                                                         jnp.sum(weighted_square_error) / num_valid_gt,
                                                         0.0)

            else:
                raise ValueError(f"Feature {feature_name} not found")
               
        penalty_losses = {}
        for penalty_name, weight in self.penalty_weights.items():
            penalty_losses[penalty_name] = weight * penalties[penalty_name]

        loss = sum(feature_losses.values()) + sum(penalty_losses.values()) + estim_loss

        aux = {
            'eap': eap,
            'mem_current_per_area': mem_current_per_area,
            'mem_voltage': mem_voltage,
            'eap_pre_filter': eap_pre_filter,
            'full_params': full_params,
            'opt_params': opt_params,
            'params': params,
            'feature_losses': feature_losses,
            'penalty_losses': penalty_losses,
            'penalties': penalties,
            'eap_features': eap_features,
            'estim_loss': estim_loss
        }

        return loss, aux
    
    # Define fixed ordering for GradNorm-compatible loss
    EI_FEATURE_ORDER = ['diffusion_peaks', 'ei_widths_di_so', 'ei_widths_so_po', 
                        'potassium_peaks', 'sodium_peaks', 'sodium_velocity']
    PENALTY_ORDER = ['axon_rise', 'first_order_continuity', 'second_order_continuity']
    
    def loss_v2(self, opt_params, estim_data, ei_weights, penalty_weights, estim_weight):
        """
        Loss function with weights as explicit arguments for GradNorm compatibility.
        
        Args:
            opt_params: Optimization parameters
            estim_data: Electrical stimulation data
            ei_weights: JAX array of shape (6,) - weights for EI features in EI_FEATURE_ORDER
            penalty_weights: JAX array of shape (3,) - weights for penalties in PENALTY_ORDER
            estim_weight: Scalar weight for estim loss
            
        Returns:
            loss: Total weighted loss
            aux: Dictionary with intermediate values and per-feature unweighted losses
        """
        params = self.forward_sigmoid_params_v1(opt_params)
        full_params = self.expand_params_v1(params)
        eap, mem_current_per_area, mem_voltage, eap_pre_filter = self.jited_predict_ei(full_params)

        if self._training_estim_enabled:
            estim_loss_unweighted = self.estim_loss(full_params, estim_data)
            estim_loss = estim_weight * estim_loss_unweighted
        else:
            estim_loss_unweighted = 0.0
            estim_loss = 0.0

        eap_features = self.get_ei_features(eap, training_only=True)
        penalties = self.get_penalties(params)
        
        feature_losses = {}  # Weighted losses
        feature_losses_unweighted = {}  # Unweighted losses for GradNorm
        
        # NOTE: loss_v2 does NOT use self.loss_elec_weightings (important_elecs weighting)
        # GradNorm handles all loss balancing dynamically
        for i, feature_name in enumerate(self.EI_FEATURE_ORDER):
            weight = ei_weights[i]
            if not self._is_ei_feature_enabled(feature_name):
                feature_losses[feature_name] = 0.0
                feature_losses_unweighted[feature_name] = 0.0
                continue
            
            if feature_name in ['sodium_peaks', 'diffusion_peaks', 'potassium_peaks']:
                square_error = jnp.square(eap_features[feature_name] - self.gt_features[feature_name])
                # Uniform weighting - no important_elecs bias
                unweighted_loss = jnp.mean(square_error)
                feature_losses_unweighted[feature_name] = unweighted_loss
                feature_losses[feature_name] = weight * unweighted_loss

            elif feature_name in ['ei_widths_so_po', 'ei_widths_di_so']:
                width_gt, valid_elec_bool_gt = self.gt_features[feature_name]
                width_pred, valid_elec_bool_pred = eap_features[feature_name]
                valid_elec_bool = valid_elec_bool_gt & valid_elec_bool_pred
                square_error = jnp.where(valid_elec_bool, jnp.square(width_pred - width_gt), 0.0)
                # Uniform weighting - no important_elecs bias
                # Handle case where no valid electrodes (empty override) - loss is 0
                num_valid_gt = jnp.sum(valid_elec_bool_gt)
                unweighted_loss = jnp.where(num_valid_gt > 0, 
                                           jnp.sum(square_error) / num_valid_gt, 
                                           0.0)
                feature_losses_unweighted[feature_name] = unweighted_loss
                feature_losses[feature_name] = weight * unweighted_loss

            elif feature_name in ['sodium_velocity']:
                velocity_gt, _, valid_elec_pairs_bool_gt = self.gt_features[feature_name]
                velocity_pred, _, valid_elec_pairs_bool_pred = eap_features[feature_name]
                # Don't apply proximity mask if override was used (override pairs may be far apart)
                if self.velocity_elecs_override_used:
                    valid_elec_pairs_bool = valid_elec_pairs_bool_gt & valid_elec_pairs_bool_pred
                else:
                    valid_elec_pairs_bool = valid_elec_pairs_bool_gt & valid_elec_pairs_bool_pred & self.elec_proximity_mask
                square_error = jnp.where(valid_elec_pairs_bool, jnp.square(velocity_pred - velocity_gt), 0.0)
                # Uniform weighting - no important_elecs bias
                # Handle case where no valid pairs (division by zero protection)
                num_valid_gt = jnp.sum(valid_elec_pairs_bool_gt)
                unweighted_loss = jnp.where(num_valid_gt > 0,
                                           jnp.sum(square_error) / num_valid_gt,
                                           0.0)
                feature_losses_unweighted[feature_name] = unweighted_loss
                feature_losses[feature_name] = weight * unweighted_loss

        penalty_losses = {}
        penalty_losses_unweighted = {}
        for i, penalty_name in enumerate(self.PENALTY_ORDER):
            weight = penalty_weights[i]
            unweighted_loss = penalties[penalty_name]
            penalty_losses_unweighted[penalty_name] = unweighted_loss
            penalty_losses[penalty_name] = weight * unweighted_loss

        loss = sum(feature_losses.values()) + sum(penalty_losses.values()) + estim_loss

        aux = {
            'eap': eap,
            'mem_current_per_area': mem_current_per_area,
            'mem_voltage': mem_voltage,
            'eap_pre_filter': eap_pre_filter,
            'full_params': full_params,
            'opt_params': opt_params,
            'params': params,
            'feature_losses': feature_losses,
            'feature_losses_unweighted': feature_losses_unweighted,
            'penalty_losses': penalty_losses,
            'penalty_losses_unweighted': penalty_losses_unweighted,
            'penalties': penalties,
            'eap_features': eap_features,
            'estim_loss': estim_loss,
            'estim_loss_unweighted': estim_loss_unweighted,
        }

        return loss, aux
    
    def get_default_weights_v2(self):
        """
        Get default weights as JAX arrays in the order expected by loss_v2.
        
        Returns:
            ei_weights: shape (6,) array
            penalty_weights: shape (3,) array  
            estim_weight: scalar
        """
        ei_weights = jnp.array([
            self.ei_feature_weights.get(name, 0.0) for name in self.EI_FEATURE_ORDER
        ])
        penalty_weights = jnp.array([
            self.penalty_weights.get(name, 0.0) for name in self.PENALTY_ORDER
        ])
        estim_weight = self.estim_loss_weight
        return ei_weights, penalty_weights, estim_weight
    
    def loss_full(self, full_params):
        eap, mem_current_per_area, mem_voltage = self.jited_predict_ei(full_params)
        loss = jnp.mean(jnp.abs(eap))
        aux = {'eap': eap,
               'mem_current_per_area': mem_current_per_area,
               'mem_voltage': mem_voltage,
               'full_params': full_params}
        return loss, aux
    
    def get_default_kish_2023_values_dict(self):
        """
        Get the default values for the parameters.
        """
        return {
            'HH_gNa_dendrite': 0.06, 'HH_gNa_soma': 0.06, 'HH_gNa_AH': 0.15, 'HH_gNa_SOCB': 0.42, 'HH_gNa_NR': 0.15, 'HH_gNa_axon': 0.1,
            'HH_gK_dendrite': 0.035, 'HH_gK_soma': 0.035, 'HH_gK_AH': 0.09, 'HH_gK_SOCB': 0.25, 'HH_gK_NR': 0.09, 'HH_gK_axon': 0.05,
            'radius_dendrite': 0.5, 'radius_soma': 10.0, 'radius_AH': 1.5, 'radius_NR': 0.4, 'radius_axon': 0.5, 'radius_scale_factor': 1.0,
            'extra_resistivity_ohm_cm': 1000.0
        }
    
    def get_param_bounds_v1(self):
        """
        Get biophysically realistic bounds for the parameters centered around the default values.
        No zeros allowed.
        """
        param_bounds = {
            'HH_gNa_dendrite': (0.03, 0.09), 'HH_gNa_soma': (0.03, 0.18), 'HH_gNa_AH': (0.075, 0.3), 'HH_gNa_SOCB': (0.25, 0.75), 'HH_gNa_NR': (0.09, 0.3), 'HH_gNa_axon': (0.05, 0.25),
            'HH_gK_dendrite': (0.015, 0.055), 'HH_gK_soma': (0.015, 0.11), 'HH_gK_AH': (0.045, 0.2), 'HH_gK_SOCB': (0.15, 0.5), 'HH_gK_NR': (0.045, 0.2), 'HH_gK_axon': (0.025, 0.1),
            'radius_dendrite': (0.45, 0.55), 'radius_soma': (8.0, 12.0), 'radius_AH': (1.3, 1.7), 'radius_NR': (0.3, 0.5), 'radius_axon': (0.4, 0.6),
            'radius_scale_factor': (0.4, 2.5), 
            'extra_resistivity_ohm_cm': (500.0, 1500.0)
        }

        # Assert that the midpoint of each bound matches the default value
        defaults = self.get_default_kish_2023_values_dict()

        for param_name, default_val in defaults.items():
            #midpoint = (param_bounds[param_name][0] + param_bounds[param_name][1]) / 2
            #assert abs(midpoint - default_val) < 1e-6, \
            #   f"Midpoint {midpoint} of bounds for {param_name} does not match default value {default_val}"
            assert param_bounds[param_name][0] < default_val < param_bounds[param_name][1], \
                f"Default value {default_val} for {param_name} is not within bounds {param_bounds[param_name]}"

        return param_bounds
    
    def get_initial_guess_z_v1(self):
        """
        Get the initial guess for the z-coordinates of the axon notch locations.
        """
        min_bounds, max_bounds = self.get_depth_bounds_v1()
        return (min_bounds + max_bounds) / 2.0

    def get_depth_bounds_v1(self):
        """
        Get bounds for the location of the axon notch locations.
        """
        general_bounds = {
            'soma_depth': (-40.0, -20.0),
            'axon_depth': (-20.0, -3.0)
        }
        total_depth_bounds = (general_bounds['soma_depth'][0], general_bounds['axon_depth'][1])

        # Calculate how many control points are within 200um of first point
        points_close_to_soma = int(self.soma_smooth_dist_um / self.control_point_spacing) + 1
        
        # Create arrays of min and max bounds
        min_bounds = jnp.ones(self.num_control_points) * general_bounds['axon_depth'][0]
        max_bounds = jnp.ones(self.num_control_points) * general_bounds['axon_depth'][1]
        
        # Set first point bounds to soma depth
        min_bounds = min_bounds.at[0].set(general_bounds['soma_depth'][0])
        max_bounds = max_bounds.at[0].set(general_bounds['soma_depth'][1])
        
        # points within 100um of first point are set to the total depth bounds
        #for i in range(1, points_close_to_soma):
        #    t = i / points_close_to_soma
        #    min_bounds = min_bounds.at[i].set(total_depth_bounds[0])
        #    max_bounds = max_bounds.at[i].set(total_depth_bounds[1])
            
        return (min_bounds, max_bounds)

    def get_random_params(self, 
                          param_bounds: Dict[str, tuple],
                          num_control_points: int,
                          control_point_spacing: float,
                          xy_bounds: float,
                          soma_depth_bounds: tuple,
                          axon_depth_bounds: tuple,
                          seed: int,
                          percentile_sample_and_clip: float = 0.85):
        """
        Generate random parameters for synthetic RGC model.
        
        Args:
            param_bounds: Dictionary of parameter bounds.
            num_control_points: Number of control points for axon spline.
            control_point_spacing: Spacing between control points in um.
            xy_bounds: Half-width (radius) of square for XY coordinate perturbations. Control points are uniformly distributed in a square centered at each control point center with half-width xy_bounds.
            soma_depth_bounds: Tuple (min, max) for soma depth (Z coordinate of first control point).
            axon_depth_bounds: Tuple (min, max) for axon depth (Z coordinates of subsequent control points).
            seed: Random seed for reproducibility.
            percentile_sample_and_clip: Fraction to clip sampling range around mean (0.85 means sample within 85% of range). Default: 0.85.
        
        Returns:
            Dictionary of random parameters in the format expected by expand_params_v1.
        """
        # Assert xy_bounds is positive
        assert xy_bounds > 0, f"xy_bounds must be positive, got {xy_bounds}"
        assert len(soma_depth_bounds) == 2 and soma_depth_bounds[0] <= soma_depth_bounds[1], \
            f"soma_depth_bounds must be a tuple (min, max) with min <= max, got {soma_depth_bounds}"
        assert len(axon_depth_bounds) == 2 and axon_depth_bounds[0] <= axon_depth_bounds[1], \
            f"axon_depth_bounds must be a tuple (min, max) with min <= max, got {axon_depth_bounds}"
        
        # Assert param_bounds are valid
        for param_name, bounds in param_bounds.items():
            assert len(bounds) == 2, f"param_bounds[{param_name}] must be a tuple (min, max), got {bounds}"
            assert bounds[0] <= bounds[1], \
                f"param_bounds[{param_name}] must have min <= max, got {bounds}"
        
        np.random.seed(seed)
        
        # Sample biophysical parameters
        random_params = {}
        for param_name, (lower, upper) in param_bounds.items():
            mean = (lower + upper) / 2.0
            range_size = upper - lower
            clip_coeff = abs(percentile_sample_and_clip - 0.5)
            sampled_val = np.random.uniform(
                mean - clip_coeff * range_size,
                mean + clip_coeff * range_size
            )
            random_params[param_name] = jnp.array(sampled_val)
        
        # Generate control points with specified structure
        # Centers: (0, 0, soma_depth), (spacing, 0, axon_depth), (2*spacing, 0, axon_depth), ...
        soma_depth_center = (soma_depth_bounds[0] + soma_depth_bounds[1]) / 2.0
        axon_depth_center = (axon_depth_bounds[0] + axon_depth_bounds[1]) / 2.0
        
        # Sample soma depth
        soma_depth_range = soma_depth_bounds[1] - soma_depth_bounds[0]
        soma_depth = np.random.uniform(
            soma_depth_center - clip_coeff * soma_depth_range,
            soma_depth_center + clip_coeff * soma_depth_range
        )
        
        # Sample axon depth (will be used for all points after first)
        axon_depth_range = axon_depth_bounds[1] - axon_depth_bounds[0]
        axon_depth = np.random.uniform(
            axon_depth_center - clip_coeff * axon_depth_range,
            axon_depth_center + clip_coeff * axon_depth_range
        )
        
        # Generate control point centers
        # Centers: (0, 0, soma_depth), (spacing, 0, axon_depth), (2*spacing, 0, axon_depth), ...
        # Control points are uniformly distributed in a square centered at each center,
        # with half-width (radius) given by xy_bounds
        control_point_centers_xy = []
        control_point_centers_z = []
        
        for i in range(num_control_points):
            # X coordinate center: i * spacing
            x_center = i * control_point_spacing
            # Y coordinate center: 0 (straight line in Y)
            y_center = 0.0
            
            # Add uniform random perturbation in square [-xy_bounds, xy_bounds] for both X and Y
            x_perturb = np.random.uniform(-xy_bounds, xy_bounds)
            y_perturb = np.random.uniform(-xy_bounds, xy_bounds)
            
            x_final = x_center + x_perturb
            y_final = y_center + y_perturb
            
            control_point_centers_xy.append([x_final, y_final])
            
            # Z coordinate: soma_depth for first point, axon_depth for rest
            if i == 0:
                z_center = soma_depth
                z_bounds = soma_depth_bounds
            else:
                z_center = axon_depth
                z_bounds = axon_depth_bounds
            
            # Add small random perturbation to Z, then clip to bounds
            z_range = z_bounds[1] - z_bounds[0]
            z_perturb_scale = z_range * 0.1  # 10% of depth range
            z_perturb = np.random.uniform(-z_perturb_scale, z_perturb_scale)
            z_final = np.clip(z_center + z_perturb, z_bounds[0], z_bounds[1])
            
            control_point_centers_z.append(z_final)
        
        # Convert to JAX arrays
        control_point_centers_xy = jnp.array(control_point_centers_xy)
        control_point_centers_z = jnp.array(control_point_centers_z)
        
        # Clip Z coordinates
        control_point_centers_z = control_point_centers_z.at[0].set(
            jnp.clip(control_point_centers_z[0], soma_depth_bounds[0], soma_depth_bounds[1])
        )
        control_point_centers_z = control_point_centers_z.at[1:].set(
            jnp.clip(control_point_centers_z[1:], axon_depth_bounds[0], axon_depth_bounds[1])
        )
        
        random_params['axon_spline_control_points_xy'] = control_point_centers_xy
        random_params['axon_spline_control_points_z'] = control_point_centers_z
        
        return random_params

    def predict_ei_synthetic(self, params: Dict[str, jnp.ndarray], seed=0):
        """
        Predict extracellular potentials from synthetic parameters.
        
        This function takes parameters in the format expected by expand_params_v1,
        expands them to full_params, and computes the EI.
        
        Args:
            params: Dictionary of parameters (same format as input to expand_params_v1).
                   Should include:
                   - HH_gNa_*, HH_gK_* for each region
                   - radius_* for each region
                   - radius_scale_factor
                   - extra_resistivity_ohm_cm
                   - axon_spline_control_points_xy
                   - axon_spline_control_points_z
        
        Returns:
            Tuple of (eap_mV, aux) where:
            - eap_mV is the filtered extracellular potential (time x electrodes)
            - aux is a dictionary containing:
              - mem_current_per_area: membrane current per area
              - mem_voltage: membrane voltage
              - eap_mV_pre_filter: EI before filtering
              - full_params: expanded full parameters
              - compartment_locations: 3D locations of all compartments
              - control_points_xy: XY control points
              - control_points_z: Z control points
        """
        # Expand parameters to full format
        full_params = self.expand_params_v1(params)
        
        # Predict EI using existing machinery
        eap_mV, mem_current_per_area, mem_voltage, eap_mV_pre_filter = self.predict_ei(full_params, seed=seed)
        
        # Create aux dictionary with all useful information
        aux = {
            'mem_current_per_area': mem_current_per_area,
            'mem_voltage': mem_voltage,
            'eap_mV_pre_filter': eap_mV_pre_filter,
            'full_params': full_params,
            'compartment_locations': full_params['compartment_locations'],
            'control_points_xy': params['axon_spline_control_points_xy'],
            'control_points_z': params['axon_spline_control_points_z'],
        }
        
        return eap_mV, aux

    def xtra_multielec_stim(self, full_params: Dict[str, jnp.ndarray], 
                electrode_intensities_uA: jnp.ndarray, baseline_stimulus: jnp.ndarray) -> jnp.ndarray:
        #baseline stimulus is a 1d array, this code forces all electrodes to have the same baseline stimulus
        #scaled at different intensities at each electrode
        #multiply by 2.0 since rgc model assumes half plane current source
        extracellular_potential_mV_per_elec = 2.0 * point_source_stim_potential_mV_per_uA(
            extra_resistivity_ohm_cm=full_params['extra_resistivity_ohm_cm'], 
            source_locations_um=self.electrode_locations_um, 
            record_locations_um=full_params['compartment_locations']
        ) #shape: (n_compartments, n_elecs)

        xtra_potential_comps_mV = extracellular_potential_mV_per_elec @ electrode_intensities_uA
        axial_resistance_compartments = full_params['axial_resistivity'] * (jnp.array(full_params['length']) / (jnp.pi * full_params['radius']**2)) * 1e-2 # Mohm
        comp_potential_diffs = jnp.where(
            self.adjacency_matrix,
            xtra_potential_comps_mV[None, :] - xtra_potential_comps_mV[:, None],
            0.0
        )
        inter_comp_axial_resistance = (axial_resistance_compartments[:, None] + axial_resistance_compartments[None, :]) / 2.0 #Mohm
        intra_comp_currents = (comp_potential_diffs / inter_comp_axial_resistance) #nA
        currents_compartments = jnp.sum(intra_comp_currents, axis=1) #nA
        current_time_series = baseline_stimulus[None, :] * currents_compartments[:, None]
        data_stimuli = self.ei_cell.data_stimulate(current_time_series)

        cell_params = [{param_name: full_params[param_name]} for param_name in self.cell_params_list]
        outputs = jx.integrate(self.estim_cell, delta_t=self.ESTIM_TIME_STEP, params=cell_params, data_stimuli=data_stimuli).reshape(
            len(self.estim_cell_record_values), self.num_total_compartments, -1)
        
        v = outputs[0, :, :]
        return v.transpose()
    
    def adjust_stim_intensities(self, full_params: Dict[str, jnp.ndarray], all_elec_intensities: jnp.ndarray) -> jnp.ndarray:
        is_positive = all_elec_intensities > 0.0
        adjustment = jnp.where(is_positive, full_params['positive_current_adjustment'], full_params['negative_current_adjustment'])
        return all_elec_intensities * adjustment

    def triphasic_xtra_singleelec_stim(self, full_params: Dict[str, jnp.ndarray], elec:int, intensity:float) -> jnp.ndarray:
        intensities = jnp.zeros(self.ELECTRODE_LOCATIONS_UM.shape[0])
        intensities = intensities.at[elec].set(intensity)
        adjusted_intensities = self.adjust_stim_intensities(full_params, intensities)
        return self.xtra_multielec_stim(full_params, adjusted_intensities, 
            baseline_stimulus=self.triphasic_stimulus)
    
    def reduced_triphasic_xtra_singleelec_stim(self, full_params: Dict[str, jnp.ndarray], elec:int, intensity:float) -> jnp.ndarray:
        intensities = jnp.zeros(self.ELECTRODE_LOCATIONS_UM.shape[0])
        intensities = intensities.at[elec].set(intensity)
        adjusted_intensities = self.adjust_stim_intensities(full_params, intensities)
        return self.xtra_multielec_stim(full_params, adjusted_intensities, 
            baseline_stimulus=self.triphasic_stimulus[0:self.estim_indicator_sample+1])
    
    def _is_spike(self, v_i, indicator_sample, min_voltage_mV=0.0):
        """
        Check if a spike occurred in the evoked voltage trace.
        """
        return jnp.any(v_i[indicator_sample:, :] > min_voltage_mV)
    
    def is_spike_triphasic_xtra_multielec_stim(self, full_params: Dict[str, jnp.ndarray], 
                elecs:jnp.ndarray, intensities:jnp.ndarray) -> jnp.ndarray:
        intensities_holder = jnp.zeros(self.ELECTRODE_LOCATIONS_UM.shape[0])
        intensities_holder = intensities_holder.at[elecs].set(intensities)
        adjusted_intensities = self.adjust_stim_intensities(full_params, intensities_holder)
        v_i = self.xtra_multielec_stim(full_params, adjusted_intensities, 
            baseline_stimulus=self.triphasic_stimulus)
        return self._is_spike(v_i, indicator_sample=self.estim_indicator_sample, min_voltage_mV=0.0)
    
    def is_spike_single_elec(self, full_params:Dict[str, jnp.ndarray], 
            electrode:int, intensity:float, indicator_sample:int, min_voltage_mV=0.0):
        v_i = self.jitted_single_elec_stim(
            full_params=full_params, 
            elec=electrode, 
            intensity=intensity
        )
        spike = self._is_spike(v_i, indicator_sample=indicator_sample, min_voltage_mV=min_voltage_mV)
        return spike
    
    def is_spike_single_elec_fast(self, full_params:Dict[str, jnp.ndarray], 
            electrode:int, intensity:float, min_voltage_mV=0.0):
        """
        Fast version that simulates up to 2x indicator_sample (~440ms instead of 1500ms).
        Still ~3x faster than full simulation while giving more time for spike to propagate.
        """
        # Reuse the same intensity adjustment pattern as reduced_triphasic_xtra_singleelec_stim
        intensities = jnp.zeros(self.ELECTRODE_LOCATIONS_UM.shape[0])
        intensities = intensities.at[electrode].set(intensity)
        adjusted_intensities = self.adjust_stim_intensities(full_params, intensities)
        v_i = self.xtra_multielec_stim(
            full_params, 
            adjusted_intensities,
            baseline_stimulus=self.triphasic_stimulus[0:self.estim_indicator_sample * 2]
        )
        # Check if any compartment voltage at the last time point exceeds threshold
        return jnp.max(v_i[-1, :]) > min_voltage_mV
    
    def get_estim_single_elec_critical_voltage(self, full_params:Dict[str, jnp.ndarray], 
            electrode:int, intensity:float):
        v_i = self.jitted_reduced_single_elec_stim(
            full_params=full_params, 
            elec=electrode, 
            intensity=intensity
        )
        return jnp.max(v_i[self.estim_indicator_sample, :])

    def critical_voltage_to_probability(self, critical_voltage_mV:float, threshold_voltage_mV:float,
                is_spike_scale:float=10.0, not_spike_scale:float=5.0):
        """
        Convert a critical voltage to a probability.
        """
        adjust_factor = jnp.where(critical_voltage_mV > threshold_voltage_mV, 
                                  is_spike_scale, not_spike_scale)
        adjusted_voltage = (critical_voltage_mV - threshold_voltage_mV) / adjust_factor
        #pass adjusted voltage through a sigmoid
        return 1.0 / (1.0 + jnp.exp(-adjusted_voltage))
    
    def estim_single_elec_differentiable_probability(self,
            full_params:Dict[str, jnp.ndarray],
            electrode:int, intensity:float, threshold_voltage_mV:float,
            is_spike_scale:float=10.0, not_spike_scale:float=5.0):
        """
        Estimate the probability of a spike for a single electrode stimulus.
        This is differentiable with respect to the full_params since using sigmoid of 
        the critical voltage.
        """
        critical_voltage_mV = self.get_estim_single_elec_critical_voltage(
            full_params, electrode, intensity)
        return self.critical_voltage_to_probability(
            critical_voltage_mV, threshold_voltage_mV, 
            is_spike_scale=is_spike_scale, not_spike_scale=not_spike_scale)

    def binary_search_threshold(self, electrode:int, upper_bound:float, num_iterations:int, full_params:Dict[str, jnp.ndarray], is_positive:bool):
        """
        Binary search to find threshold for a single electrode.
        """
        def search_step(i, bounds):
            lower, upper = bounds
            mid = (lower + upper) / 2
            
            spike = self.is_spike_single_elec(
                full_params=full_params, 
                electrode=electrode, 
                intensity=jnp.where(is_positive, mid, -mid), 
                indicator_sample=self.estim_indicator_sample, 
                min_voltage_mV=0.0,
            )
            return (jnp.where(spike, lower, mid), jnp.where(spike, mid, upper))

        lower = 0.0
        upper = upper_bound
        final_bounds = jax.lax.fori_loop(0, num_iterations, search_step, (lower, upper))
        
        lower, upper = final_bounds
        amp_lower = self.get_estim_single_elec_critical_voltage(
            full_params, electrode, jnp.where(is_positive, lower, -lower))
        amp_upper = self.get_estim_single_elec_critical_voltage(
            full_params, electrode, jnp.where(is_positive, upper, -upper))

        return (jnp.where(is_positive, lower, -upper), 
                jnp.where(is_positive, upper, -lower)), (amp_lower, amp_upper)

    def binary_search_threshold_fast(self, electrode:int, upper_bound:float, num_iterations:int, full_params:Dict[str, jnp.ndarray], is_positive:bool):
        """
        Fast binary search to find threshold for a single electrode.
        Only simulates up to indicator_sample+1 (~220ms instead of 1500ms).
        """
        def search_step(i, bounds):
            lower, upper = bounds
            mid = (lower + upper) / 2
            
            spike = self.is_spike_single_elec_fast(
                full_params=full_params, 
                electrode=electrode, 
                intensity=jnp.where(is_positive, mid, -mid), 
                min_voltage_mV=0.0,
            )
            return (jnp.where(spike, lower, mid), jnp.where(spike, mid, upper))

        lower = 0.0
        upper = upper_bound
        final_bounds = jax.lax.fori_loop(0, num_iterations, search_step, (lower, upper))
        
        lower, upper = final_bounds
        # Still use get_estim_single_elec_critical_voltage for the final critical voltage
        # (it already uses reduced stim)
        amp_lower = self.get_estim_single_elec_critical_voltage(
            full_params, electrode, jnp.where(is_positive, lower, -lower))
        amp_upper = self.get_estim_single_elec_critical_voltage(
            full_params, electrode, jnp.where(is_positive, upper, -upper))

        return (jnp.where(is_positive, lower, -upper), 
                jnp.where(is_positive, upper, -lower)), (amp_lower, amp_upper)
