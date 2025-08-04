import re
import copy
import importlib
import ml_collections as mlc


def model_config(
    low_prec=False, 
    use_deepspeed_evoformer_attention=False,
):  
    
    c = copy.deepcopy(config)
    # TRAINING PRESETS
    if use_deepspeed_evoformer_attention:
        c.globals.use_deepspeed_evo_attention = True 
    if low_prec:
        c.globals.eps = 1e-4
        # c.globals.inf = 1e4

    return c


c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(64, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)
c_s_inputs = mlc.FieldReference(384 + 31 + 31 + 1, field_type=int)
c_atom = mlc.FieldReference(128, field_type=int)
c_atompair = mlc.FieldReference(16, field_type=int)
c_token = mlc.FieldReference(768, field_type=int) # 384 in the input embedder
sigma_data = mlc.FieldReference(16, field_type=float)
confidence_enabled = mlc.FieldReference(True, field_type=bool)

chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(1e9, field_type=float)
templates_enabled = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)

sampling_steps = mlc.FieldReference(200, field_type=int)

config = mlc.ConfigDict(
    {
        "globals": {
            "chunk_size": chunk_size,
            # Use DeepSpeed memory-efficient attention kernel
            "use_deepspeed_evo_attention": False,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_s": c_s,
            "c_s_inputs": c_s_inputs,
            "c_atom": c_atom,
            "c_atompair": c_atompair,
            "c_token": c_token,
            "sigma_data": sigma_data,
            "confidence_enabled": confidence_enabled,
            "eps": eps,
            "inf": inf,
        },
        
        "backbone": {
            "recycling_iters" : 3,
            "_mask_trans": True,
            "input_embedder": {
                "c_z": c_z,
                "c_s": c_s,
                "c_s_inputs": c_s_inputs,
                "c_atom": c_atom,
                "c_atompair": c_atompair,
                "c_token": 384,
                "c_ref": 3 + 1 + 128 + 1 + 4 * 64,
                "no_blocks": 3,
                "no_heads": 4,
                "window_size_row": 32,
                "window_size_col": 128,
                "r_max" :32,
                "s_max": 2,
                "inf": inf,
                "eps": eps,  # 1e-6
                "tune_chunk_size": tune_chunk_size,
            },
            "recycling_embedder": {
                "c_s": c_s,
                "c_z": c_z,
                "eps": eps,  # 1e-6,
                "inf": inf,
            },
            "template_embedder": {
                "c_z": c_z,
                "c_a": 39 + 1 + 3 + 1 + 31 + 31,
                "no_bins": 39,
                "c_t": c_t,
                "no_blocks": 2,
                "c_hidden_mul": 64,
                "c_hidden_pair_att": 16,
                "no_heads_pair": 4,
                "transition_n": 2,
                "pair_dropout": 0.25,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-6,
                "enabled": templates_enabled,
            },
            "msa": {
                "msa_embedder": {
                    "c_msa_feat": 34,
                    "c_m": c_m,
                    "c_s_inputs": c_s_inputs,
                    "msa_depth": 1024,
                },
                "msa_stack": {
                    "c_m": c_m,
                    "c_z": c_z,
                    "c_hidden_msa_att": 8,
                    "c_hidden_opm": 32,
                    "c_hidden_mul": 128,
                    "c_hidden_pair_att": 32,
                    "no_heads_msa": 8,
                    "no_heads_pair": 4,
                    "no_blocks": 4,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "inf": inf,
                    "eps": eps,  # 1e-10,
                    "tune_chunk_size": tune_chunk_size,
                },
            },
            "pairformer_stack": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "no_heads_single": 16,
                "no_heads_pair": 4,
                "no_blocks": 48,
                "transition_n": 4,
                "pair_dropout": 0.25,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
            "heads": {
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                },
                "confidence_head": {
                    "enabled": confidence_enabled,
                    "_mask_trans": True,
                    "c_z": c_z,
                    "c_s": c_s,
                    "c_s_inputs": c_s_inputs,
                    "no_bin_pae": 64,
                    "no_bin_pde": 64,
                    "no_bin_plddt": 50,
                    "min_bin": 3.25,
                    "max_bin": 50.75,
                    "no_bins": 39,
                    "max_num_atoms" : 24,
                    "eps": eps,  # 1e-6,
                    "inf": inf,
                    "pairformer_stack": {
                        "c_s": c_s,
                        "c_z": c_z,
                        "c_hidden_mul": 128,
                        "c_hidden_pair_att": 32,
                        "no_heads_single": 16,
                        "no_heads_pair": 4,
                        "no_blocks": 4,
                        "transition_n": 4,
                        "pair_dropout": 0.25,
                        "tune_chunk_size": tune_chunk_size,
                        "inf": inf,
                        "eps": eps,  # 1e-10,
                    },
                },
                },
            },
        "diffusion": {
            "window_size_row": 32,
            "window_size_col": 128,
            "sigma_data": sigma_data,
            "diffusion_conditioning": {
                "c_z": c_z,
                "c_s": c_s,
                "c_s_inputs": c_s_inputs,
                "c_fourier": 256,
                "sigma_data": sigma_data,
                "no_transitions":2 ,
                "transition_n": 2,
                "r_max": 32,
                "s_max": 2,
                "inf": inf,
                "eps": eps,  # 1e-6,
            },
            "atom_attention_encoder": {
                "c_atom": c_atom,
                "c_atompair": c_atompair,
                "c_token": c_token,
                "c_s": c_s,
                "c_z": c_z,
                "c_ref": 3 + 1 + 128 + 1 + 4 * 64,
                "no_blocks": 3,
                "no_heads": 4,
                "window_size_row": 32,
                "window_size_col": 128,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
            "diffusion_transformer": {
                "c_a": c_token,
                "c_z": c_z,
                "c_s": c_s,
                "no_blocks": 24,
                "no_heads": 16,
                "transition_n": 2,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
            "atom_attention_decoder": {
                "c_atom": c_atom,
                "c_atompair": c_atompair,
                "c_token": c_token,
                "no_blocks": 3,
                "no_heads": 4,
                "window_size_row": 32,
                "window_size_col": 128,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
        },
        "confidence_head": {
            "enabled": confidence_enabled,
            "_mask_trans": True,
            "c_z": c_z,
            "c_s": c_s,
            "c_s_inputs": c_s_inputs,
            "no_bin_pae": 64,
            "no_bin_pde": 64,
            "no_bin_plddt": 50,
            "min_bin": 3.25,
            "max_bin": 50.75,
            "no_bins": 39,
            "max_num_atoms" : 24,
            "eps": eps,  # 1e-6,
            "inf": inf,
            "pairformer_stack": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "no_heads_single": 16,
                "no_heads_pair": 4,
                "no_blocks": 4,
                "transition_n": 4,
                "pair_dropout": 0.25,
                "tune_chunk_size": tune_chunk_size,
                "inf": inf,
                "eps": eps,  # 1e-10,
            },
        },
        "sample": {
            "sigma_data": sigma_data,
            "no_sample_steps_T": sampling_steps,
            "mini_roll_out_steps_T": 20,
            "sigma_max": 160,
            "sigma_min": 4e-4,
            "rho": 7,
            "P_mean": -1.2,
            "P_std":1.5,
            "gamma_0": 0.8,
            "gamma_min": 1.0,
            "noise_scale_lambda": 1.003,
            "step_scale_eta": 1.5,
        },
    }
)
