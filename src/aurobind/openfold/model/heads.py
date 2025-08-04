# Copyright 2024 IntelliGen-AI and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from aurobind.openfold.model.primitives import Linear, LayerNorm
from aurobind.openfold.utils.precision_utils import is_fp16_enabled
from aurobind.openfold.model.pairformer import PairformerStack
from aurobind.openfold.utils.atom_token_conversion import aggregate_fn
from aurobind.openfold.utils.tensor_utils import add
from typing import Dict, Optional, Tuple
import einops
import numpy as np

class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.distogram = DistogramHead(
            **config["distogram"],
        )
        self.ligand_affinity = LigandAffinityHead(config)
        self.config = config

    def forward(self, outputs, feats):
        aux_out = {}

        distogram_logits = self.distogram(outputs["z"])
        aux_out["distogram_logits"] = distogram_logits

        head_pair_mask = feats["seq_mask"][:, None, :] * feats["seq_mask"][:, :, None]
        protein_mask = feats['is_protein']
        pair_protein_mask = protein_mask[:, None, :] * protein_mask[:, :, None]
        head_pair_mask = head_pair_mask & ~pair_protein_mask
        affinity_logits = self.ligand_affinity(
            outputs["s"],
            outputs["z"],
            single_mask=feats["seq_mask"],
            is_ligand=feats["is_ligand"],
            head_pair_mask=head_pair_mask
        )
        aux_out["affinity_logits"] = affinity_logits
        return aux_out


class LigandAffinityHead(torch.nn.Module):
    def __init__(self, config):
        """
        Args:
            c_s:
                Input channel dimension
        """
        super().__init__()
        # pairformer blocks
        self.config = config.confidence_head    
        self.c_s = self.config["c_s"]
        self.c_z = self.config["c_z"]

        self.config_pairformer_stack = self.config["pairformer_stack"]    
        self.pairformer_stack_head = PairformerStack(
            ** self.config_pairformer_stack
        )

        self.gate_ln = torch.nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        self.gate_linear = torch.nn.Linear(self.c_s, 1, bias=True)
        self.ln = torch.nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        # self.linear = torch.nn.Linear(self.c_s, 1)
        hidden_size = 512
        self.mlp = nn.Sequential(
            nn.Linear(self.c_s, hidden_size),  # First layer
            nn.ReLU(),                          # Activation
            nn.Linear(hidden_size, hidden_size), # Second layers
            nn.ReLU(),                          # Activation
            # nn.Dropout(0.5),                    # Dropout for regularization
            nn.Linear(hidden_size, 1) # Output layer
        )

        self.z_gate_ln = torch.nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_gate_linear = torch.nn.Linear(self.c_z, 1, bias=True)
        self.z_ln = torch.nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_mlp = nn.Sequential(
            nn.Linear(self.c_z, hidden_size),  # First layer
            nn.ReLU(),                          # Activation
            nn.Linear(hidden_size, hidden_size), # Second layer
            nn.ReLU(),                          # Activation
            # nn.Dropout(0.5),                    # Dropout for regularization
            nn.Linear(hidden_size, 1) # Output layer
        )

    def forward(self, s, z, single_mask=None, is_ligand=None, head_pair_mask=None, weighted=2.0):

        inplace_safe = False
        pair_mask = single_mask[..., None] * single_mask[..., None, :]
        s_update, z_update = self.pairformer_stack_head(
                s,
                z,
                single_mask=single_mask.to(dtype=s.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=None,
                use_deepspeed_evo_attention=True,
                inplace_safe=inplace_safe,
                _mask_trans=True,
            )
        s = add(s, s_update, inplace_safe)
        z = add(z, z_update, inplace_safe)

        bs = s.shape[0]
        s_dim = s.shape[-1]
        z_dim = z.shape[-1]


        mask = is_ligand.bool()
        single_mask = single_mask.bool()
        affinities = []

        for i in range(bs):

            s_batch = s[i][single_mask[i]]

            gate = self.gate_linear(self.gate_ln(s_batch)).sigmoid()
            s_affinity = (gate * self.mlp(self.ln(s_batch)))

            z_mask = pair_mask[i]
            z_mask_head = head_pair_mask[i]
            z_mask = z_mask.bool()
            z_batch = z[i][z_mask] 
            if head_pair_mask is not None:
                num_tokens = z_mask[0].sum()
            else:
                num_tokens = mask[i].sum()

            z_batch = z_batch.reshape(num_tokens, num_tokens, z_dim)
            
            z_gate = self.z_gate_linear(self.z_gate_ln(z_batch)).sigmoid()
            if head_pair_mask is not None:
                z_affinity = (z_gate * self.z_mlp(self.z_ln(z_batch))).sum(dim=-2) / z_mask_head.sum(dim=-1, keepdim=True)[:num_tokens]
            else:
                z_affinity = (z_gate * self.z_mlp(self.z_ln(z_batch))).mean(dim=-2)

            batch_affinity = s_affinity + z_affinity

            ligand_weight_ratio = weighted
            is_lig = is_ligand[i].float()
            num_ligands = is_lig.sum()
            num_proteins = num_tokens - num_ligands
            ratio = ligand_weight_ratio * (num_proteins / num_ligands)
            affinity_weight = (is_lig * ratio + (1 - is_lig))[:num_tokens]
            normalized_weight = affinity_weight / affinity_weight.sum()
            # temperature = 3.0
            # normalized_weight = torch.softmax(affinity_weight / temperature, dim=0)
            
            batch_affinity = batch_affinity * normalized_weight.unsqueeze(-1)
            affinities.append(batch_affinity.sum(dim=-2))

        affinity = torch.stack(affinities, dim=0)
        
        return affinity


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins)

    def _forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
    
    def forward(self, z): 
        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(z.float())
        else:
            return self._forward(z)

class pAEHead(nn.Module):
    def __init__(self, c_s, no_bins, **kwargs):
        super(pAEHead, self).__init__()

        self.c_s = c_s
        self.no_bins = no_bins
        self.layer_norm = LayerNorm(self.c_s)
        self.linear = Linear(self.c_s, self.no_bins)
        
    def _forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] pair embedding
        Returns:
            [*, N_res, no_bins] pAE probability distribution
        """
        # [*, N_res, no_bins]
        s = self.layer_norm(s)
        logits = self.linear(s)
        return logits
    
    def forward(self, s):
        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(s.float())
        else:
            return self._forward(s)
        
class pDEHead(nn.Module):
    def __init__(self, c_s, no_bins, **kwargs):
        super(pDEHead, self).__init__()

        self.c_s = c_s
        self.no_bins = no_bins
        self.layer_norm = LayerNorm(self.c_s)
        self.linear = Linear(self.c_s, self.no_bins)
        
    def _forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] pair embedding
        Returns:
            [*, N_res, no_bins] pDE probability distribution
        """
        # [*, N_res, no_bins]
        s = self.layer_norm(s)
        logits = self.linear(s)
        logits = logits + logits.transpose(-2, -3)
        return logits
    
    def forward(self, s):
        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(s.float())
        else:
            return self._forward(s)


class pLDDTHead(nn.Module):
    def __init__(self, c_s, max_num_atoms, no_bins, **kwargs):
        super(pLDDTHead, self).__init__()

        self.c_s = c_s
        self.no_bins = no_bins
        self.max_num_atoms = max_num_atoms
        self.layer_norm = LayerNorm(self.c_s)
        self.linear = Linear(self.c_s, max_num_atoms * self.no_bins)
        
    def _forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] pair embedding
        Returns:
            [*, N_res, no_bins] pLDDT probability distribution
        """
        # [*, N_res, no_bins]
        s = self.layer_norm(s)
        logits = self.linear(s)
        return logits
    
    def forward(self, s):
        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(s.float())
        else:
            return self._forward(s)
        
class ResolvedHead(nn.Module):
    def __init__(self, c_s, max_num_atoms, **kwargs):
        super(ResolvedHead, self).__init__()

        self.c_s = c_s
        self.layer_norm = LayerNorm(self.c_s)
        self.linear = Linear(self.c_s, max_num_atoms * 2)
        
    def _forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] pair embedding
        Returns:
            [*, N_res, 2] resolved probability distribution
        """
        # [*, N_res, 2]
        s = self.layer_norm(s)
        logits = self.linear(s)
        return logits
    
    def forward(self, s):
        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(s.float())
        else:
            return self._forward(s)
        

class ConfidenceHead(nn.Module):
    def __init__(self, config):
        super(ConfidenceHead, self).__init__()

        self.config = config.confidence_head
        self.globals = config.globals
        self.c_s_inputs = self.config["c_s_inputs"]
        self.c_s = self.config["c_s"]
        self.c_z = self.config["c_z"]
        self.inf = self.config["inf"]
        self.eps = self.config["eps"]
        self.config_pairformer_stack = self.config["pairformer_stack"]
        self.no_bin_pae = self.config["no_bin_pae"]
        self.no_bin_pde = self.config["no_bin_pde"]
        self.no_bin_plddt = self.config["no_bin_plddt"]
        self.max_bin = self.config["max_bin"]
        self.min_bin = self.config["min_bin"]
        self.no_bins = self.config["no_bins"]
        self.max_num_atoms = self.config["max_num_atoms"]
        
        self.linear_s_inputs_row = Linear(self.c_s_inputs, self.c_z, bias=False)
        self.linear_s_inputs_col = Linear(self.c_s_inputs, self.c_z, bias=False)
        
        self.linear_d = Linear(self.no_bins,self.c_z, bias=False)

        self.pairformer_stack = PairformerStack(
            ** self.config_pairformer_stack
        )
        
        self.pae_head = pAEHead(self.c_z, self.no_bin_pae)
        self.pde_head = pDEHead(self.c_z, self.no_bin_pde)
        self.plddt_head = pLDDTHead(self.c_s, self.max_num_atoms, self.no_bin_plddt)
        self.resolved_head = ResolvedHead(self.c_s, self.max_num_atoms )
        
    def forward(self, s_inputs ,s ,z ,x_pred ,batch):
        """
        Args:
            s_inputs:
                [*, N_res, C_s_inputs] single representation from input features
            s:
                [*, N_res, C_s] single representation from pairformer trunk
            z:
                [*, N_res, N_res, C_z] pair representation from pairformer trunk
            x_pred:
                [*, N_atom, 3] predicted atom positions
            batch:
                batch dictionary

        Returns:
            Dictionary of confidence 
        """
        batch_size = s_inputs.shape[0]
        diffusion_batch_size = x_pred.shape[0] // batch_size
        # batch size and diffusion batch size can not be larger than one at the same time
        assert (batch_size == 1) or (diffusion_batch_size == 1), "batch size and diffusion batch size can not be larger than one at the same time"
        
        inplace_safe = not (self.training or torch.is_grad_enabled())
        inplace_safe = False
        
        single_mask = batch["seq_mask"]
        pair_mask = single_mask[..., None] * single_mask[..., None, :]
        atom_pseudo_beta_index = batch["atom_pseudo_beta_index"]
        pseudo_beta_mask = batch["pseudo_beta_mask"]
        pred_dense_atom_mask = batch["pred_dense_atom_mask"]
        
        s_inputs_row = self.linear_s_inputs_row(s_inputs)
        s_inputs_col = self.linear_s_inputs_col(s_inputs)
        z = z + s_inputs_row.unsqueeze(-2) + s_inputs_col.unsqueeze(-3)
        # get the representative atom
        x_pred_rep = torch.zeros(s.shape[:-1]+(3,), device=x_pred.device, dtype=x_pred.dtype)
        x_pred_rep = torch.gather(einops.rearrange(x_pred,"b l n d -> b (l n) d "), dim = 1,
                                  index=einops.repeat(atom_pseudo_beta_index, "b l -> (b n) l 3", n = diffusion_batch_size)) \
            * pseudo_beta_mask[..., None] * single_mask[..., None]
        
        
        lower_breaks = torch.linspace(
            self.min_bin, self.max_bin, self.no_bins, device=z.device
        )
        lower_breaks = lower_breaks ** 2
        
        upper_breaks = torch.cat([lower_breaks[1:], torch.tensor(self.inf, device= z.device).reshape(1)], dim=0)
        
        dist2 = torch.sum((x_pred_rep[..., None, :] - x_pred_rep[..., None, :, :]) ** 2, dim=-1, keepdims=True) * pair_mask.unsqueeze(-1)

        dgram = (dist2 > lower_breaks).to(z.dtype) * (dist2 <= upper_breaks).to(z.dtype) * pair_mask.unsqueeze(-1)

        z = add(
            z,
            self.linear_d(dgram),
            False,
        )
        if diffusion_batch_size > 1:
            s_output = torch.zeros_like(s)
            s_output = einops.repeat(s_output, "b ... -> (b n) ...", n = diffusion_batch_size)
            z_output = torch.zeros_like(z)
            for j in range(diffusion_batch_size):
                s_output[j:j+1], z_output[j:j+1] = self.pairformer_stack(
                    s,
                    z[j:j+1],
                    single_mask=single_mask.to(dtype=s.dtype),
                    pair_mask=pair_mask.to(dtype=z.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )
        else:
            s_output, z_output = self.pairformer_stack(
                s,
                z,
                single_mask=single_mask.to(dtype=s.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )
        s = s_output
        z = z_output
        
        p_pae = self.pae_head(z)
        p_pde = self.pde_head(z)
        p_plddt = self.plddt_head(s).view(*s.shape[:-1], self.max_num_atoms, self.no_bin_plddt)
        p_resolved = self.resolved_head(s).view(*s.shape[:-1], self.max_num_atoms, 2)
        confidence_out = {}

        confidence_out['pae_logits'] = p_pae
        confidence_out['pde_logits'] = p_pde
        confidence_out['plddt_logits'] = p_plddt
        confidence_out['resolved_logits'] = p_resolved
        plddt = torch.zeros(p_plddt.shape[:-1], device=p_plddt.device)
        pae = torch.zeros(p_pae.shape[:-1], device=p_pae.device)
        pde = torch.zeros(p_pde.shape[:-1], device=p_pde.device)
        ptm = torch.zeros(batch_size * diffusion_batch_size, device=p_pae.device)
        iptm = torch.zeros(batch_size * diffusion_batch_size, device=p_pae.device)
        
        for i in range(batch_size):
            for j in range(diffusion_batch_size):
                index = i * diffusion_batch_size + j
                [aggregated_p_plddt], reverse_fn = aggregate_fn([p_plddt[index:index+1]], pred_dense_atom_mask)
                aggregated_plddt = compute_plddt(aggregated_p_plddt)
                [plddt[index:index+1]] = reverse_fn([aggregated_plddt])
                pae[index:index+1] = compute_predicted_aligned_error(p_pae[index:index+1])["predicted_aligned_error"]
                pde[index:index+1] = compute_predicted_distance_error(p_pde[index:index+1])["predicted_distance_error"]
                ptm[index:index+1] = compute_tm(p_pae[index:index+1],residue_weights=batch['frame_mask'][i:i+1] * single_mask[i:i+1])
                iptm[index:index+1] = compute_tm(p_pae[index:index+1],residue_weights=batch['frame_mask'][i:i+1] * single_mask[i:i+1], interface=True, asym_id=batch["asym_id"][i:i+1])
        confidence_out['plddt'] = plddt
        confidence_out['pae'] = pae
        confidence_out['pde'] = pde
        confidence_out['ptm'] = ptm
        confidence_out['iptm'] = iptm
        
        confidence_out['p_pae'] = p_pae
        return confidence_out
        


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt



def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_error(
    confidence_breaks: torch.Tensor,
    error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(confidence_breaks)
    return (
        torch.sum(error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_error(
        confidence_breaks=boundaries,
        error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }
    
def compute_predicted_distance_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes distance confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      distance_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        distance error probabilities over bins for each residue pair.
      predicted_distance_error: [*, num_res, num_res] the expected distance distance
        error for each pair of residues.
      max_predicted_distance_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    distance_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_distance_error,
        max_predicted_distance_error,
    ) = _calculate_expected_error(
        confidence_breaks=boundaries,
        error_probs=distance_confidence_probs,
    )

    return {
        "distance_confidence_probs": distance_confidence_probs,
        "predicted_distance_error": predicted_distance_error,
        "max_predicted_distance_error": max_predicted_distance_error,
    }

def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[..., None, :] * residue_weights[..., :, None]
    )
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = torch.argmax(weighted, dim=-1)
    argmax_unsqueeze = argmax.unsqueeze(-1)
    selected_alignments = torch.gather(per_alignment, 1, argmax_unsqueeze)
    return selected_alignments.squeeze(-1)
