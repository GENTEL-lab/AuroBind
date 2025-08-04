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

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Tuple, Optional

from aurobind.openfold.model.primitives import Linear, LayerNorm
from aurobind.openfold.model.pairformer import PairStack
from aurobind.openfold.model.diffusion import AtomAttentionEncoder, RelativePositionEncoding
import einops
from aurobind.openfold.utils.tensor_utils import add

    
class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 1 (part of), 2, 3
    """

    def __init__(
        self,
        c_z: int,
        c_s: int,
        c_s_inputs: int,
        c_atom: int,
        c_atompair: int,
        c_token: int,
        c_ref : int,
        no_blocks : int,
        no_heads : int,
        window_size_row  : int,
        window_size_col  : int,
        r_max: int,
        s_max: int,
        inf: float,
        eps: float,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        """
        Args:

            c_z:
                Pair embedding dimension
            c_s:
                Single embedding dimension
            c_s_inputs:
                Single inputs embedding dimension
            c_atom:
                Atom embedding dimension
            c_atompair:
                Atom pair embedding dimension
            c_token:
                Token embedding dimension
            c_ref:
                Reference initial embedding dimension
            no_blocks:
                Number of blocks in AtomAttentionEncoder
            no_heads:
                Number of heads in AtomAttentionEncoder
            window_size_row:
                Window size for AtomAttentionEncoder
            window_size_col:
                Window size for AtomAttentionEncoder
            r_max:
                Maximum relative index on token level
            s_max:
                Maximum relative index on chain level
        """
        super(InputEmbedder, self).__init__()

        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom = c_atom,
            c_atompair = c_atompair,
            c_token = c_token,
            c_s = None,
            c_z = None,
            c_ref =  c_ref,
            no_blocks = no_blocks,
            no_heads = no_heads,
            window_size_row = window_size_row,
            window_size_col = window_size_col,
            initial = True,
            inf=inf,
            eps=eps,
            tune_chunk_size = tune_chunk_size,
        )
            

        self.linear_s_inputs = Linear(c_s_inputs,c_s, bias=False) 
        self.linear_z_i = Linear(c_s_inputs, c_z, bias=False) 
        self.linear_z_j = Linear(c_s_inputs, c_z, bias=False) 

        self.linear_token_bonds = Linear(1, c_z,bias=False) 

        self.relative_position_encoding = RelativePositionEncoding(
            c_z = c_z,
            r_max = r_max,
            s_max = s_max,
        )

    def forward(
        self, 
        batch,
        chunk_size=None,
        use_deepspeed_evo_attention=False, 
        inplace_safe=False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Batch dictionary
            chunk_size:
                Chunk size for attention
            use_deepspeed_evo_attention:
                Whether to use EvoFormer attention
            inplace_safe:
                Whether to perform inplace operations
        Returns:
            s:
                [*, N_token, C_s] Single embedding
            z:
                [*, N_token, N_token, C_z] Pair embedding
        """
        # Algo 2 line 1
        a,_,_,_ = self.atom_attention_encoder(
            ref_pos = batch["ref_pos"],
            ref_charge = batch["ref_charge"],
            ref_mask = batch["ref_mask"],
            ref_element = batch["ref_element"],
            ref_atom_name_chars = batch["ref_atom_name_chars"],
            ref_space_uid = batch["ref_space_uid"],
            atom_mask = batch["aggregated_pred_dense_atom_mask"],
            s_trunk = None,
            z = None,
            r = None,
            molecule_atom_lens = batch["molecule_atom_lens"],
            chunk_size = chunk_size,
            use_deepspeed_evo_attention = use_deepspeed_evo_attention,
            inplace_safe = inplace_safe,
        )
        
        # Algo 2 line 2
        s_inputs = torch.cat([batch['aatype'],batch['profile'],batch['deletion_mean'].unsqueeze(-1),a],dim=-1)
        
        # Algo 1 line 2
        s = self.linear_s_inputs(s_inputs)

        # Algo 1 line 3
        z = self.linear_z_i(s_inputs).unsqueeze(-2) + self.linear_z_j(s_inputs).unsqueeze(-3)
        
        # Algo 1 line 4
        z = add(z, self.relative_position_encoding(
            batch["asym_id"],
            batch["residue_index"],
            batch["entity_id"],
            batch["token_index"],
            batch["sym_id"],
            dtype=z.dtype,
        ), inplace=inplace_safe)
        
        # Algo 1 line 5
        z = add(z, self.linear_token_bonds(batch["token_bonds"].unsqueeze(-1)), inplace=inplace_safe)
        
        return s, z, s_inputs

class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 1 (part of)
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single channel dimension
            c_z:
                Pair embedding channel dimension
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf

        self.linear_z = Linear(c_z, c_z,bias=False) 
        self.layer_norm_z = LayerNorm(self.c_z) 
        self.linear_s = Linear(c_s, c_s,bias=False) 
        self.layer_norm_s = LayerNorm(self.c_s)
        
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, N_token, C_s] single embedding
            z:
                [*, N_token, N_token, C_z] pair embedding
        Returns:
            s:
                [*, N_token, C_m] Single embedding update
            z:
                [*, N_token, N_token, C_z] pair embedding update
        """

        # Algo 1 line 8
        # [*, N, N, C_z]
        z_update = self.linear_z(self.layer_norm_z(z))
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z
            
        
        # Algo 1 line 11
        # [*, N, C_s]
        s_update = self.linear_s(self.layer_norm_s(s))
        if(inplace_safe):
            s.copy_(s_update)
            s_update = s

        return s_update, z_update


class MSAEmbedder(nn.Module):
    """
    Embeds MSA sequences.

    Implements Algorithm 8 (part of)
    """
    def __init__(
        self,
        c_msa_feat: int,
        c_m: int,
        c_s_inputs: int,
        msa_depth: int,
        **kwargs,
    ):
        """
        Args:
            c_msa_feat:
                MSA feature dimension
            c_m:
                MSA embedding dimension
            c_s_inputs:
                Single embedding dimension
            msa_depth:
                Depth of MSA sampling
        """
        super(MSAEmbedder, self).__init__()

        self.c_msa_feat = c_msa_feat
        self.c_m = c_m
        self.c_s_inputs = c_s_inputs
        self.msa_depth = msa_depth

        self.linear_mf = Linear(c_msa_feat, c_m,bias=False)
        self.linear_s_inputs = Linear(c_s_inputs, c_m,bias=False)

    def forward(self, batch,s_inputs,msa_mask,inplace_safe=False):
        """
        Args:
            batch:
                Batch dictionary
            s_inputs:
                [* , N_token, C_s_inputs] Single inputs embedding
            msa_mask:
                [* , N_msa, N_token] MSA mask
            inplace_safe:
                Whether to perform inplace operations
        Returns:
            m:
                [* , N_msa, N_token, C_m] MSA embedding
        """
        # Algo 8 line 1
        mf = torch.cat(
            [batch["msa"],batch["has_deletion"].unsqueeze(-1),batch["deletion_value"].unsqueeze(-1)],
            dim=-1)
        # Algo 8 line 2
        # SampleRandomWithoutReplacement
        max_depth = min(self.msa_depth, mf.shape[1])
        random_num_select = [max_depth for _ in batch['num_alignments']]
        
        # Generate random indices for selection
        random_indices = [
            torch.randperm(num_align, device=mf.device)[:num_select]
            for num_align, num_select in zip(batch['num_alignments'], random_num_select)
        ]        
        indices = torch.stack(
            [F.pad(indice, (0, max_depth - len(indice)), value=-1) for indice in random_indices], dim=0
        ) 
        
        valid_mask = indices >= 0
        indices = indices.masked_fill(~valid_mask, 0)
        
        gathered_mf = torch.gather(
            mf,
            1,
            einops.repeat(indices, 'b n -> b n m d', m=mf.shape[-2], d=mf.shape[-1])
        )  
        mf = gathered_mf.masked_fill(~valid_mask.unsqueeze(-1).unsqueeze(-1), 0)

        gathered_msa_mask = torch.gather(
            msa_mask,
            1,
            einops.repeat(indices, 'b n -> b n m', m=msa_mask.shape[-1])
        ) 
        msa_mask = gathered_msa_mask.masked_fill(~valid_mask.unsqueeze(-1), 0)
        
        # Algo 8 line 3
        m = self.linear_mf(mf)
        # Algo 8 line 4
        m = add(m, self.linear_s_inputs(s_inputs).unsqueeze(dim = -3), inplace=inplace_safe )
        
        return m, msa_mask

 
class TemplateEmbedder(nn.Module):
    """
    Embeds template features.
    
    Implements Algorithm 16
    """
    def __init__(
        self, 
        c_z: int,
        c_t: int,
        c_a: int,
        no_bins: int,
        no_blocks,
        c_hidden_mul: int,
        c_hidden_pair_att : int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
        **kwargs,
        ):
        """
        Args:
            c_z:
                Pair embedding dimension
            c_t:
                Template embedding dimension
            c_a:
                Intermediate embedding dimension
            no_bins:
                Number of template distogram bins
            no_blocks:
                Number of Pairformer blocks
            c_hidden_mul:
                Multiplier for hidden dimension
            c_hidden_pair_att:
                Hidden dimension for pair attention
            no_heads_pair:
                Number of heads for pair attention
            transition_n:
                Number of transitions
            pair_dropout:
                Pair dropout
            inf:
                Large value for masking
            eps:
                Epsilon value for normalization
        
        """
        
        super(TemplateEmbedder, self).__init__()
        
        self.c_z = c_z
        self.c_t = c_t
        self.c_a = c_a
        self.no_bins = no_bins
        self.eps = eps

        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, c_t,bias=False)
        self.linear_d = Linear(no_bins, c_t,bias=False)
        self.linear_d_mask = Linear(1, c_t,bias=False)
        self.linear_aatype_col = Linear(31, c_t,bias=False)
        self.linear_aatype_row = Linear(31, c_t,bias=False)
        
        self.linear_unit_vec_x = Linear(1, c_t,bias=False)
        self.linear_unit_vec_y = Linear(1, c_t,bias=False)
        self.linear_unit_vec_z = Linear(1, c_t,bias=False)
        
        self.linear_bb_mask = Linear(1, c_t,bias=False)
        
        
        self.pairformer_stack = nn.ModuleList([]) 
        for _ in range(no_blocks):
            block = PairStack(
                c_z = c_t,
                c_hidden_mul= c_hidden_mul,
                c_hidden_pair_att= c_hidden_pair_att,
                no_heads_pair= no_heads_pair,
                transition_n= transition_n,
                pair_dropout= pair_dropout,
                inf= inf,
                eps= eps,
            )

            self.pairformer_stack.append(block)

        
        self.layer_norm_t = LayerNorm(c_t)
        self.linear_o = Linear(c_t, c_z,bias=False)
    
    def forward_layers(self,
                       z,
                       pair_mask,
                       chunk_size,
                       use_deepspeed_evo_attention,
                       inplace_safe,
                       _mask_trans,):
        for block in self.pairformer_stack:
            z = block(
                z = z,
                pair_mask = pair_mask,
                chunk_size = chunk_size,
                use_deepspeed_evo_attention = use_deepspeed_evo_attention,
                inplace_safe = inplace_safe,
                _mask_trans = _mask_trans,
                _attn_chunk_size = chunk_size)
        return z
                 
    def forward(self, 
        batch, 
        z,
        pair_mask, 
        chunk_size,
        _mask_trans=True,
        use_deepspeed_evo_attention=False,
        inplace_safe=False
    ):
        """
        Args:
            batch:
                Batch dictionary
            z:
                [*, N_token, N_token, C_z] Pair embedding
            pair_mask:
                [*, N_token, N_token] Pair mask
            chunk_size:
                Chunk size for attention
            _mask_trans:
                Whether to mask transitions
            use_deepspeed_evo_attention:
                Whether to use EvoFormer attention
            inplace_safe:
                Whether to perform inplace operations
        Returns:
            u:
                [*, N_token, N_token, C_z] Pair embedding update
        
        """
        n_templ = batch["template_aatype"].shape[-3]
        asym_id = batch["asym_id"]
        b_same_chain = (asym_id[..., None] == asym_id[..., None, :])
        
        # Algo 16 line 1
        templ_backbone_frame_mask = batch["template_backbone_frame_mask"]
        templ_backbone_frame_mask_2d = templ_backbone_frame_mask.unsqueeze(-1) * templ_backbone_frame_mask.unsqueeze(-2)
        templ_backbone_frame_mask_2d = templ_backbone_frame_mask_2d * b_same_chain.unsqueeze(-3)
        
        # Algo 16 line 2
        templ_pseudo_beta_mask = batch["template_pseudo_beta_mask"]
        templ_pseudo_beta_mask_2d = templ_pseudo_beta_mask.unsqueeze(-1) * templ_pseudo_beta_mask.unsqueeze(-2)
        templ_pseudo_beta_mask_2d = templ_pseudo_beta_mask_2d * b_same_chain.unsqueeze(-3)
        
        templ_distogram = batch["template_distogram"]
        templ_unit_vector = batch["template_unit_vector"]
        unit_vec_x = templ_unit_vector[..., 0:1]
        unit_vec_y = templ_unit_vector[..., 1:2]
        unit_vec_z = templ_unit_vector[..., 2:3]
        templ_aatype = batch["template_aatype"]

        # Algo 16 line 6
        u = torch.zeros(z.shape[:-1] + (self.c_t,), device=z.device, dtype=z.dtype)
        
        # Algo 16 line 7
        for t in range(n_templ):
            
            v = self.linear_d(templ_distogram[:,t])
            v = add(v, self.linear_d_mask(templ_pseudo_beta_mask_2d[:,t].unsqueeze(-1)), inplace=inplace_safe)
            v = add(v, self.linear_aatype_col(templ_aatype[:,t].unsqueeze(-3)), inplace=inplace_safe)
            v = add(v, self.linear_aatype_row(templ_aatype[:,t].unsqueeze(-2)), inplace=inplace_safe)
            v = add(v, self.linear_unit_vec_x(unit_vec_x[:,t]), inplace=inplace_safe)
            v = add(v, self.linear_unit_vec_y(unit_vec_y[:,t]), inplace=inplace_safe)
            v = add(v, self.linear_unit_vec_z(unit_vec_z[:,t]), inplace=inplace_safe)
            v = add(v, self.linear_bb_mask(templ_backbone_frame_mask_2d[:,t].unsqueeze(-1)), inplace=inplace_safe)
            
            v = add(v, self.linear_z(self.layer_norm_z(z)), inplace=inplace_safe)
        
            # Algo 16 line 9
            v = self.forward_layers(
                        z= v, 
                        pair_mask = pair_mask, 
                        chunk_size =chunk_size, 
                        use_deepspeed_evo_attention =use_deepspeed_evo_attention, 
                        inplace_safe=inplace_safe, 
                        _mask_trans=_mask_trans)
            # Algo 16 line 10
            u = add(u,
                    self.layer_norm_t(v),
                    inplace=inplace_safe)
            

        # [*, N, N, C_z]
        # Algo 16 line 12
        u = u / ( n_templ + self.eps)
        # Algo 16 line 13
        u = torch.nn.functional.relu(u)
        u = self.linear_o(u)

        return u
