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


from typing import Optional, Tuple, Sequence,NamedTuple, Tuple
import einops
import torch
import torch.nn as nn

from aurobind.openfold.utils.tensor_utils import add, one_hot
from aurobind.openfold.utils.chunk_utils import chunk_layer, ChunkSizeTuner
from aurobind.openfold.model.primitives import Linear, LayerNorm
from aurobind.openfold.model.pairformer import Transition, AttentionPairBias, AdaLN
from aurobind.openfold.utils.atom_token_conversion import repeat_consecutive_with_lens, concat_previous_and_later_windows, pad_at_dim, mean_pool_with_lens
    
class ConditionedTransitionBlock(nn.Module):
    """
    SwiGLU transition block with adaptive layernorm
    
    Implements Algorithm 25
    """
    def __init__(self, c_a, c_s, transition_n):
        """
        Args:
            c_a:
                input channel dimension
            c_s:
                condition channel dimension 
            transition_n:
                Factor by which c is multiplied to obtain hidden channel
                dimension
        """
        super(ConditionedTransitionBlock, self).__init__()

        self.c_a = c_a
        self.c_s = c_s
        self.transition_n = transition_n

        self.adaptive_layer_norm = AdaLN(c_a, c_s)
        
        self.linear_a = Linear(self.c_a, self.transition_n * self.c_a * 2,bias=False)
        self.swish = nn.SiLU()
        
        self.linear_s = Linear(self.c_s, self.c_a,bias=True)
        self.linear_b = Linear(self.c_a * self.transition_n , self.c_a,bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def _transition(self, a,s, mask):

        a = self.adaptive_layer_norm(a, s)
        a1, a2 = torch.chunk(self.linear_a(a), chunks=2, dim = -1)
        b = self.swish(a1) * a2
        a = self.sigmoid(self.linear_s(s)) * self.linear_b(b)
        a = a * mask
        return a

    @torch.jit.ignore
    def _chunk(self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"a": a, "s":s,"mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(a.shape[:-2]),
        )

    def forward(self, 
        a: torch.Tensor, 
        s: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N_token, C] or [*, N_atom, C] embedding
            s:
                [*, N_token, C] or [*, N_atom, C] condition
        Returns:
            [*, N_token, C] or [*, N_atom, C] embedding update
        """
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        # [*, N_token, 1] or [*, N_atom, 1]
        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            a = self._chunk(a, s, mask, chunk_size)
        else:
            a = self._transition(a=a,s =s, mask=mask)

        return a

    
class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()

        weight = torch.randn(1, c)
        bias = torch.randn(c)
        
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(
        self,
        t,
    ):  

        rand_proj = torch.einsum('...i, ij -> ...j', t, self.weight)
        rand_proj = rand_proj + self.bias
        return torch.cos(2 * torch.pi * rand_proj)


class DiffusionTransformerStack(nn.Module):
    """
    The Diffusion Transformer Stack
    
    Implements Algorithm 23
    """

    def __init__(
        self,
        no_blocks,
        no_heads,
        c_a,
        c_s,
        c_z,
        window_size_row = None,
        window_size_col = None,
        transition_n = 2,
        eps = 1e-8,
        inf = 1e9,
        tune_chunk_size: bool = False,
        **kwargs
    ):
        """
        Args:
            no_blocks:
                Number of transformer blocks
            no_heads:
                Number of attention heads
            c_a:
                Single channel dimension
            c_s:
                Single channel dimension
            c_z:
                Pair channel dimension
            window_size_row:
                Window size for local attention
            window_size_col:
                Window size for local attention
            transition_n:
                Transition expansion factor
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super().__init__()

        self.layer_norm_z = LayerNorm(c_z, bias=False)


        self.blocks = nn.ModuleList([])

        for _ in range(no_blocks):
            block = DiffusionTransformerBlock(
                c_a,
                c_s,
                c_z,
                no_heads,
                window_size_row = window_size_row,
                window_size_col = window_size_col,
                transition_n = transition_n,
                eps = eps,
                inf = inf,
            )
            self.blocks.append(block)

        
        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if(tune_chunk_size):
            self.chunk_size_tuner = ChunkSizeTuner()

    def forward(self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N_token, C] or [*, N_atom, C] embedding
            s:
                [*, , N_token, C_m] single embedding
            z:
                [*, N_token, N_token, C_z] pair embedding
            mask:
                [*, N_token] single mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
            inplace_safe:
                Whether to perform in-place operations
        Returns:
            z:
                [*, N_token, N_token, C_z] pair embedding
            s:
                [*, N_token, C_s] single embedding
        """ 
        z = self.layer_norm_z(z)
        
        for block in self.blocks:
            a,s,z = block(a,s,z,
                          mask=mask,
                          chunk_size=chunk_size,
                          use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                          inplace_safe=inplace_safe)
    
        return a


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        c_a,
        c_s,
        c_z,
        no_heads,
        window_size_row = None,
        window_size_col = None,
        transition_n = 2,
        eps = 1e-8,
        inf = 1e9,
        **kwargs
    ):

        super(DiffusionTransformerBlock, self).__init__()


        self.window_size_row = window_size_row
        self.window_size_col = window_size_col

        self.attention_pair_bias = AttentionPairBias(
                    c_a = c_a,
                    c_s = c_s,
                    c_z = c_z,
                    c_hidden = c_a // no_heads,
                    no_heads = no_heads,
                    window_size_row = window_size_row,
                    window_size_col = window_size_col,
                    conditioned = True,
                    use_layer_norm_z=False,
                    inf = inf,
                    eps = eps,
                )

        self.single_transition = ConditionedTransitionBlock(
                    c_a = c_a,
                    c_s = c_s,
                    transition_n = transition_n
                )

    def forward(
        self,
        a: Optional[torch.Tensor],
        s: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
        _attn_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.attention_pair_bias(
            a=a,
            s=s,
            z=z,
            single_mask = mask,
            pair_mask = None,
            chunk_size = chunk_size,
            use_deepspeed_evo_attention = use_deepspeed_evo_attention,
            inplace_safe = inplace_safe,
        )

        a = b + a

        ff_out = self.single_transition(
            a,
            s,
            mask,
            chunk_size = chunk_size,
        )

        a = a + ff_out
        
        return a,s,z


class AtomToTokenPooler(nn.Module):
    def __init__(
        self,
        c_atom,
        c_token,
    ):
        """
        Args:
            c_atom:
                Atom channel dimension
            c_token:
                Token channel dimension
        """
        super().__init__()

        self.linear_q = Linear(c_atom, c_token,bias=False)
        self.relu = nn.ReLU()

    def forward(
        self,
        q,
        atom_mask,
        molecule_atom_lens
    ):
        """
        Args:
            q:
                [*, N_atom, C_atom] atom features
            atom_mask:
                [*, N_atom] atom mask
            molecule_atom_lens:
                [*,] molecule atom lengths
        Returns:
            [*, N_token, C_token] token features
        """
        q = self.relu(self.linear_q(q))
        q = q * atom_mask.unsqueeze(-1)
        tokens = mean_pool_with_lens(q, atom_mask,molecule_atom_lens)
        return tokens


class RelativePositionEncoding(nn.Module):
    """
    Embeds relative positions.
    
    Implements Algorithm 3
    """
    def __init__(self, c_z: int, r_max: int, s_max: int, mapping=True, **kwargs):
        """
        Args:
            c_z:
                Pair embedding dimension
            r_max:
                Maximum relative index on token level
            s_max:
                Maximum relative index on chain level
        """
        super(RelativePositionEncoding, self).__init__()

        self.c_z = c_z
        self.r_max = r_max
        self.s_max = s_max

        self.no_bins = (
            2 * r_max + 2 +
            2 * r_max + 2 +
            1 +
            2 * s_max + 2
        )
        self.mapping = mapping
        
        if mapping:
            self.linear_relpos = Linear(self.no_bins, c_z,bias=False)
        
    def forward(self, asym_id,residue_index,entity_id,token_index,sym_id, dtype):

        # Algo 3 Line 1
        b_same_chain = (asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2))
        
        # Algo 3 Line 2
        b_same_residue = (residue_index.unsqueeze(-1) == residue_index.unsqueeze(-2))
        
        # Algo 3 Line 3
        b_same_entity = (entity_id.unsqueeze(-1) == entity_id.unsqueeze(-2))

        
        # Algo 3 line 4
        d_residue = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)

        d_residue = torch.clamp(
            d_residue + self.r_max, 0, 2 * self.r_max
        )
        
        d_residue = torch.where(
            b_same_chain,
            d_residue,
            (2 * self.r_max + 1) * torch.ones_like(d_residue)
        )

  
        boundaries = torch.arange(
            start=0, end=2 * self.r_max + 2, device=d_residue.device
        )
        
        # Algo 3 line 5
        a_rel_pos = one_hot(
            d_residue,
            boundaries,
        )
        
        # Algo 3 line 6
        d_token = token_index.unsqueeze(-1) - token_index.unsqueeze(-2)
        
        d_token = torch.clamp(
            d_token + self.r_max, 0, 2 * self.r_max
        )
        
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            (2 * self.r_max + 1) * torch.ones_like(d_token)
        )
        
        # Algo 3 line 7
        a_rel_token = one_hot(
            d_token,
            boundaries,
        )

        # Algo 3 line 8
        d_chain = sym_id.unsqueeze(-1) - sym_id.unsqueeze(-2)
        
        d_chain = torch.clamp(
            d_chain + self.s_max, 0, 2 * self.s_max
        )
        
        d_chain = torch.where(
            b_same_entity,
            d_chain,
            (2 * self.s_max + 1) * torch.ones_like(d_chain)
        )
        
        boundaries = torch.arange(
            start=0, end=2 * self.s_max +2, device=d_chain.device
        )
        # Algo 3 line 9
        a_rel_chain = one_hot(
            d_chain,
            boundaries,
        )
        
        # Algo 3 line 10
        rel_feat = torch.cat([a_rel_pos,a_rel_token,b_same_entity.unsqueeze(-1),a_rel_chain],dim=-1).to(
            dtype
        )

        # Algo 3 line 11
        if self.mapping:
            return self.linear_relpos(rel_feat)
        else:
            return rel_feat

class DiffusionConditioning(nn.Module):
    """
    Turn input features and single, pair representation from pairformer trunk into conditions for diffusion
    
    Implements Algorithm 21
    """
    def __init__(
        self,
        c_s,
        c_z,
        c_s_inputs,
        c_fourier,
        sigma_data,
        no_transitions,
        transition_n,
        r_max,
        s_max,
        eps,
        **kwargs
    ):
        """
        Args:
            c_s:
                Single channel dimension
            c_z:
                Pair channel dimension
            c_s_inputs:
                Single channel dimension of the input features
            c_fourier:
                Fourier embedding channel dimension
            sigma_data:
                Data noise
            no_transitions:
                Number of transitions
            transition_n:
                Transition expansion factor
            r_max:
                Maximum token distance for relative position encoding
            s_max:
                Maximum chain distance for relative position encoding
        """
        super().__init__()
        self.relative_positions_encoding = RelativePositionEncoding(
            c_z = c_z,
            r_max=r_max,
            s_max=s_max,
            mapping=False
            )
        self.no_bins = (
            2 * r_max + 2 +
            2 * r_max + 2 +
            1 +
            2 * s_max + 2
        )
        self.layer_norm_z = LayerNorm(c_z + self.no_bins, bias=False)
        self.linear_z = Linear(c_z + self.no_bins, c_z, bias = False)
        
        self.pair_transitions = nn.ModuleList([])
        for _ in range(no_transitions):
            transition = Transition(c= c_z, transition_n = transition_n)
            self.pair_transitions.append(transition)
            
        self.layer_norm_s = LayerNorm(c_s + c_s_inputs, bias=False)
        self.linear_s = Linear(c_s + c_s_inputs, c_s,bias = False)
        
        self.sigma_data = sigma_data
        self.fourier_embedding = FourierEmbedding(c_fourier)
        self.layer_norm_f = LayerNorm(c_fourier, bias=False)
        self.linear_f = Linear(c_fourier, c_s, bias = False)
        
        self.single_transitions = nn.ModuleList([])
        for _ in range(no_transitions):
            transition = Transition(c= c_s, transition_n = transition_n)
            self.single_transitions.append(transition)
            
        self.eps = eps
            
    def forward(
        self,
        asym_id,
        residue_index,
        entity_id,
        token_index,
        sym_id,
        t,
        s_trunk,
        s_inputs,
        z_trunk,
        inplace_safe=False,
    ):
        """
        Args:
            asym_id:
                [*, N_token] asymmetry id
            residue_index:
                [*, N_token] residue index
            entity_id:
                [*, N_token] entity id
            token_index:
                [*, N_token] token index
            sym_id:
                [*, N_token] symmetry id
            t:
                noisy level at the current diffusion step
            s_trunk:
                [*, N_token, C_s] single representation from pairformer trunk
            s_inputs:
                [*, N_token, C_s_inputs] single representation from input features
            z_trunk:
                [*, N_token, N_token, C_z] pair representation from pairformer trunk
            inplace_safe:
                whether to do in place operations
        """
        relative_position_encodings = self.relative_positions_encoding(asym_id, residue_index, entity_id, token_index, sym_id, dtype = z_trunk.dtype)
        z = torch.cat((z_trunk, relative_position_encodings), dim = -1)
        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        
        for transition in self.pair_transitions:
            z = add(
                z,
                transition(z),
                inplace=inplace_safe,
            )
            
        s = torch.cat((s_trunk, s_inputs), dim = -1)
        s = self.layer_norm_s(s)
        s = self.linear_s(s)

        n = self.fourier_embedding(0.25 * torch.log((t / self.sigma_data).clamp(min = self.eps)))
        n = self.layer_norm_f(n)
        # Broadcast manually if batch size is not 1
        s = einops.repeat(s, "b ... -> (b n) ..." ,n = n.shape[0] // s.shape[0]) if (s.shape[0] != 1 and n.shape[0] != s.shape[0]) else s
        
        s = add(
            s,
            self.linear_f(n),
            inplace=False, 
        )
        
        for transition in self.single_transitions:
            s = add(
                s,
                transition(s),
                inplace=inplace_safe,
            )
            
        return s, z
    
class AtomAttentionEncoder(nn.Module):
    """
    Encode the atom features into tokens with local attention
    
    Implements Algorithm 5
    """
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        c_ref,
        no_blocks,
        no_heads,
        window_size_row,
        window_size_col,
        initial = False,
        eps = 1e-8,
        inf = 1e9,
        tune_chunk_size: bool = False,
        **kwargs
    ):
        """
        Args:
            c_atom:
                Atom channel dimension
            c_atompair:
                Atom pair channel dimension
            c_token:
                Token channel dimension
            c_s:
                Single channel dimension
            c_z:
                Pair channel dimension
            c_ref:
                Reference input channel dimension
            no_blocks:
                Number of transformer blocks
            no_heads:
                Number of attention heads
            window_size_row:
                Window size for local attention
            window_size_col:
                Window size for local attention
            initial:
                Whether this is the initial atom attention encoder in the input embedder
        """
        super().__init__()
        self.initial = initial
        self.linear_ref_pos = Linear(3,c_atom,bias = False)
        self.linear_ref_atom_name_chars = Linear(4 * 64,c_atom,bias = False)
        self.linear_ref_element = Linear(128,c_atom ,bias = False)
        self.linear_ref_charge = Linear(1,c_atom, bias = False)
        self.linear_ref_mask = Linear(1,c_atom, bias = False)
        
        self.linear_d = Linear(3, c_atompair, bias = False)
            # A5 Line 5
        self.linear_d_inv = Linear(1,c_atompair, bias = False)
            # A5 Line 6
        self.linear_v = Linear(1,c_atompair, bias= False)
            # A5 Line 9
        self.window_size_row = window_size_row
        self.window_size_col = window_size_col
        
        if self.initial==False:
            self.layer_norm_s  = LayerNorm(c_s, bias=False)
            self.linear_s = Linear(c_s, c_atom, bias=False)

                # A5 Line 10
            self.layer_norm_z = LayerNorm(c_z, bias=False)
            self.linear_z = Linear(c_z, c_atompair, bias=False)
        
                # A5 Line 11
            self.linear_r = Linear(3, c_atom, bias=False)
            # A5 Line 13
        self.linear_c_row = Linear(c_atom, c_atompair, bias=False)

        self.linear_c_col =  Linear(c_atom, c_atompair, bias=False)
        
        self.linear_mlp_p_1 = Linear(c_atompair, c_atompair,bias=False)
        self.linear_mlp_p_2 = Linear(c_atompair, c_atompair,bias=False)
        self.linear_mlp_p_3 = Linear(c_atompair, c_atompair,bias=False)
        
            # A5 Line 15
        self.atom_transformer = DiffusionTransformerStack(
            c_a = c_atom,
            c_s = c_atom,
            c_z = c_atompair,
            window_size_row = window_size_row,
            window_size_col = window_size_col,
            no_blocks = no_blocks,
            no_heads = no_heads,
            eps = eps,
            inf = inf,
            tune_chunk_size = tune_chunk_size
        )
            # A5 Line 16
        self.pool_q = AtomToTokenPooler(
            c_atom = c_atom,
            c_token = c_token
        )
        
        self.relu = nn.ReLU()
    def forward(
        self,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_element,
        ref_atom_name_chars,
        ref_space_uid,
        atom_mask,
        s_trunk,
        z,
        r,
        molecule_atom_lens,
        chunk_size=None,
        use_deepspeed_evo_attention=False,
        inplace_safe=False,
    ):
        """
        Args:
            ref_pos:
                [*, N_atom, 3] ref atom positions
            ref_charge:
                [*, N_atom] ref atom charges
            ref_mask:
                [*, N_atom] ref atom mask
            ref_element:
                [*, N_atom, 128] ref atom element
            ref_atom_name_chars:
                [*, N_atom, 4, 64] atom name characters
            ref_space_uid:
                [*, N_atom] ref space uid
            atom_mask:
                [*, N_atom] atom mask
            s_trunk:
                [*, N_token, C_s] single representation from pairformer trunk
            z:
                [*, N_token, N_token, C_z] pair representation from pairformer trunk after conditioning
            r:
                [*, N_atom, 3] noisy atom positions
            molecule_atom_lens:
                [*,] molecule atom lengths
            chunk_size:
                chunk size for attention
            use_deepspeed_evo_attention:
                whether to use deepspeed evo attention
            inplace_safe:
                whether to do in place
        Returns:
            a:
                [*, N_token, C_token] token features
            q_skip:
                [*, N_atom, C_atom] atom features
            c_skip:
                [*, N_atom, C_atom] conditioned atom features
            p_skip:
                [*, N_atom, N_atom, C_atompair] atom pair features
        """
        window_size_row = self.window_size_row
        window_size_col = self.window_size_col
        if window_size_row  // 2 != window_size_col // 8:
            raise ValueError("The window sizes should be compatible")
        batch_size, atom_seq_len = ref_mask.shape

        padding_needed_row = (window_size_row - (atom_seq_len % window_size_row)) % window_size_row
        # ATOM ATTENTION ENCODER, A20 Line 3
        # A5 Line 1
        c = self.linear_ref_pos(ref_pos)
        c = add(c,self.linear_ref_mask(ref_mask.unsqueeze(-1).to(c.dtype)),inplace=inplace_safe)
        c = add(c,self.linear_ref_element(ref_element),inplace=inplace_safe)
        c = add(c,self.linear_ref_charge(torch.asinh(ref_charge.unsqueeze(-1))),inplace=inplace_safe)
        c = add(c,self.linear_ref_atom_name_chars(ref_atom_name_chars.view(*ref_atom_name_chars.shape[:-2],-1)),inplace=inplace_safe)
        c = c * ref_mask.unsqueeze(-1)
        
        # A5 Line 7
        if self.initial == False:
            seq_len = s_trunk.shape[-2]
            # A5 Line 9
            c_trunk = self.linear_s(self.layer_norm_s(s_trunk))
            c = add(c, repeat_consecutive_with_lens(c_trunk, molecule_atom_lens), inplace=inplace_safe)

        # A5 Line 13
        p_cond_row = self.linear_c_row(self.relu(c)) # [b, l, c]
        p_cond_col = self.linear_c_col(self.relu(c)) # [b, l, c]
        p_cond_row = pad_at_dim(p_cond_row, (0, padding_needed_row), value = 0., dim = -2)
        p_cond_col = pad_at_dim(p_cond_col, (0, padding_needed_row), value = 0., dim = -2)
        p_cond_row = einops.rearrange(p_cond_row, 'b (n w) d -> b n w d', w = window_size_row)
        p_cond_col = einops.rearrange(p_cond_col, 'b (n w) d -> b n w d', w = window_size_col // 8)
        p_cond_col = concat_previous_and_later_windows(p_cond_col, dim_seq = -3, dim_window = -2)
        
        p = p_cond_row.unsqueeze(-2) + p_cond_col.unsqueeze(-3)

        if self.initial == False:
            # A5 Line 10
            p_trunk = self.linear_z(self.layer_norm_z(z))
            indices = torch.arange(seq_len, device = p_trunk.device)
            indices = einops.repeat(indices, 'n -> b n', b = batch_size)
            indices = repeat_consecutive_with_lens(indices.unsqueeze(-1), molecule_atom_lens).squeeze(-1)
            
            indices_row = pad_at_dim(indices, (0, padding_needed_row), value = 0, dim = -1)
            indices_col = pad_at_dim(indices, (0, padding_needed_row), value = 0, dim = -1)
            
            indices_row = einops.rearrange(indices_row, 'b (n w) -> b n w', w = window_size_row)
            indices_col = einops.rearrange(indices_col, 'b (n w) -> b n w', w = window_size_col // 8)
            
            indices_col = concat_previous_and_later_windows(indices_col, dim_seq = -2, dim_window = -1)

            row_indices = indices_row.unsqueeze(-1)
            col_indices = indices_col.unsqueeze(-2)
            row_indices, col_indices = torch.broadcast_tensors(row_indices, col_indices)
            p = add(p,p_trunk[einops.rearrange(torch.arange(p_trunk.shape[0]), "n -> n 1 1 1"), row_indices, col_indices], inplace=inplace_safe)
        
        # A5 Line 2-6
        if padding_needed_row > 0:
            ref_pos_row = pad_at_dim(ref_pos, (0, padding_needed_row), value = 0., dim = -2) 
            ref_space_uid_row = pad_at_dim(ref_space_uid, (0, padding_needed_row), value = 0., dim = -1)
            ref_pos_col = pad_at_dim(ref_pos, (0, padding_needed_row), value = 0., dim = -2)
            ref_space_uid_col = pad_at_dim(ref_space_uid, (0, padding_needed_row), value = 0., dim = -1)
        else:
            ref_pos_row = ref_pos
            ref_space_uid_row = ref_space_uid
            ref_pos_col = ref_pos
            ref_space_uid_col = ref_space_uid
            
        ref_pos_row = einops.rearrange(ref_pos_row, 'b (n w) d -> b n w d', w = window_size_row)
        ref_pos_col = einops.rearrange(ref_pos_col, 'b (n w) d -> b n w d', w = window_size_col // 8)
        
        ref_pos_col = concat_previous_and_later_windows(ref_pos_col, dim_seq = -3, dim_window = -2)
        
        ref_space_uid_row = einops.rearrange(ref_space_uid_row, 'b (n w) -> b n w', w = window_size_row)
        ref_space_uid_col = einops.rearrange(ref_space_uid_col, 'b (n w) -> b n w', w = window_size_col // 8)
        
        ref_space_uid_col = concat_previous_and_later_windows(ref_space_uid_col, dim_seq = -2, dim_window = -1)
        
        d = ref_pos_row.unsqueeze(-2) - ref_pos_col.unsqueeze(-3)
        d_inv = 1 / (1+torch.linalg.norm(d,ord=2,dim = -1 ,keepdim=True)**2 )
        
        v = (ref_space_uid_row.unsqueeze(-1) == ref_space_uid_col.unsqueeze(-2)).to(d.dtype).unsqueeze(-1)
        
        # A5 Line 4
        p = add(p, self.linear_d(d), inplace=inplace_safe)
        # A5 Line 5
        p = add(p,self.linear_d_inv(d_inv),inplace=inplace_safe)
        # A5 Line 6
        p = add(p,self.linear_v(v),inplace=inplace_safe)
        p = v * p

        # A5 Line 14
        p2 = self.linear_mlp_p_1(self.relu(p))
        p2 = self.linear_mlp_p_2(self.relu(p2))
        p = add(p, self.linear_mlp_p_3(self.relu(p2)), inplace=inplace_safe)
        
        if self.initial == False:
            p = einops.repeat(p, "b ... -> (b n) ...", n = r.shape[0] // p.shape[0]) if (p.shape[0] != 1 and r.shape[0] != p.shape[0]) else p
            c = einops.repeat(c, "b ... -> (b n) ...", n = r.shape[0] // c.shape[0]) if (c.shape[0] != 1 and r.shape[0] != c.shape[0]) else c
            
        if self.initial == False:
            seq_len = s_trunk.shape[-2]
            q =  self.linear_r(r)
            q = add(q, c, inplace=False)
        else:
            q = c.clone()


        # A5 Line 15
        q = self.atom_transformer(
            a = q,
            s = c,
            z = p,
            mask = atom_mask,
            chunk_size = chunk_size,
            use_deepspeed_evo_attention = use_deepspeed_evo_attention,
            inplace_safe = inplace_safe,
        )
        
        # A5 Line 16
        a = self.pool_q(
            q = q,
            atom_mask = einops.repeat(atom_mask, 'b ... -> (b n) ...', n = q.shape[0] // atom_mask.shape[0]) if (q.shape[0] != atom_mask.shape[0]) else atom_mask,
            molecule_atom_lens = einops.repeat(molecule_atom_lens, 'b ... -> (b n) ...', n = q.shape[0] // molecule_atom_lens.shape[0]) if (q.shape[0] != molecule_atom_lens.shape[0]) else molecule_atom_lens
        )
        
        q_skip,c_skip,p_skip = q,c,p
        
        return a, q_skip, c_skip, p_skip

class AtomAttentionDecoder(nn.Module):
    """
    Decode the token features into atoms with local attention
    
    Implements Algorithm 6
    """
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        no_blocks,
        no_heads,
        window_size_row,
        window_size_col,
        eps = 1e-8,
        inf = 1e9,
        tune_chunk_size: bool = False,
        **kwargs
    ):
        """
        Args:
            c_atom:
                Atom channel dimension
            c_atompair:
                Atom pair channel dimension
            c_token:
                Token channel dimension
            no_blocks:
                Number of transformer blocks
            no_heads:
                Number of attention heads
            window_size_row:
                Window size for local attention
            window_size_col:
                Window size for local attention
        """
        super().__init__()
        # A6 Line 1
        self.linear_a = Linear(c_token, c_atom, bias=False)
        
        # A6 Line 2
        self.atom_transformer = DiffusionTransformerStack(
            c_a = c_atom,
            c_s = c_atom,
            c_z = c_atompair,
            window_size_row = window_size_row,
            window_size_col = window_size_col,
            no_blocks = no_blocks,
            no_heads = no_heads,
            eps = eps,
            inf = inf,
            tune_chunk_size = tune_chunk_size,
        )
        
        # A6 Line 3
        self.layer_norm_q = LayerNorm(c_atom, bias=False)
        self.linear_q = Linear(c_atom, 3, bias=False)

    def forward(
        self,
        a,
        q_skip,
        c_skip,
        p_skip,
        atom_mask,
        molecule_atom_lens,
        chunk_size,
        use_deepspeed_evo_attention,
        inplace_safe,
    ):
        """
        Args:
            a:
                [*, N_token, C_token] token features
            q_skip:
                [*, N_atom, C_atom] atom features from atom attention encoder
            c_skip:
                [*, N_atom, C_atom] conditioned atom features from atom attention encoder
            p_skip:
                [*, N_atom, N_atom, C_atompair] atom pair features from atom attention encoder
            atom_mask:
                [*, N_atom] atom mask
            molecule_atom_lens:
                [*,] molecule atom lengths
            chunk_size:
                chunk size for attention
            use_deepspeed_evo_attention:
                whether to use deepspeed evo attention
            inplace_safe:
                whether to do in place operations
        Returns:
            [*, N_atom, 3] atom position update
        """
        # ATOM ATTENTION DECODER, A20 Line 7
        # A6 Line 1
        a = self.linear_a(a)
        a = repeat_consecutive_with_lens(a, molecule_atom_lens)
        q = a + q_skip
        q = q * atom_mask.unsqueeze(-1)
        # A6 Line 2
        q = self.atom_transformer(
            a = q,
            s = c_skip,
            z = p_skip,
            mask = atom_mask,
            chunk_size = chunk_size,
            use_deepspeed_evo_attention = use_deepspeed_evo_attention,
            inplace_safe = inplace_safe
        )
        # A6 Line 3
        r_update = self.linear_q(self.layer_norm_q(q))
        return r_update
    
class DiffusionModule(nn.Module):
    """
    The Diffusion Module
    
    Implements Algorithm 20
    """

    def __init__(
        self,
        config,
    ):  
        """
        Args:
            config:
                configuration
        """
        super().__init__()
        self.config = config.diffusion
        self.globals = config.globals
        self.window_size_row = self.config.window_size_row
        self.window_size_col = self.config.window_size_col
        self.sigma_data = self.config.sigma_data
        self.c_z = self.globals.c_z
        self.c_s = self.globals.c_s
        self.c_token = self.globals.c_token

        # A20 Line 1
            # A21
        self.diffusion_conditioning = DiffusionConditioning(
            **self.config.diffusion_conditioning,
        )

        # A20 Line 3
            # A5
        self.atom_attention_encoder = AtomAttentionEncoder(
            **self.config.atom_attention_encoder,
        )

        # A20 Line 4
        self.proj_s = nn.Sequential(
            LayerNorm(self.c_s, bias=False),
            Linear(self.c_s, self.c_token,bias=False)
        )
        
        # A20 Line 5
        self.diffusion_transformer = DiffusionTransformerStack(
           **self.config.diffusion_transformer, 
        )
        # A20 Line 6
        self.layer_norm_a = LayerNorm(self.c_token, bias=False)

        # A20 Line 7
            # A6
        self.atom_attention_decoder = AtomAttentionDecoder(
            **self.config.atom_attention_decoder,
        )


    def forward(
        self,
        r_noisy,
        t,
        batch,
        s_inputs,
        s_trunk,
        z_trunk,
    ):
        """
        Args:
            r_noisy:
                [*, N_atom, 3] noisy atom positions
            t:
                noisy level at the current diffusion step
            batch:
                batch dictionary
            s_inputs:
                [*, N_token, C_s_inputs] single representation from input features
            s_trunk:
                [*, N_token, C_s] single representation from pairformer trunk
            z_trunk:
                [*, N_token, N_token, C_z] pair representation from pairformer trunk
        Returns:
            [*, N_atom, 3] denoised atom positions
        """
        
        
        inplace_safe = not (self.training or torch.is_grad_enabled())
        inplace_safe = False
        
        ref_mask = batch["ref_mask"]
        atom_mask = batch["aggregated_pred_dense_atom_mask"]
        token_mask = batch["seq_mask"]
        molecule_atom_lens = batch["molecule_atom_lens"]
        
        # Broadcast manually if batch size is larger than 1
        atom_mask = einops.repeat(atom_mask, 'b ... -> (b n) ...', n = r_noisy.shape[0] // atom_mask.shape[0]) if (atom_mask.shape[0] != 1 and r_noisy.shape[0] != atom_mask.shape[0]) else atom_mask
        token_mask = einops.repeat(token_mask, 'b ... -> (b n) ...', n = r_noisy.shape[0] // token_mask.shape[0]) if (token_mask.shape[0] != 1 and r_noisy.shape[0] != token_mask.shape[0]) else token_mask
        
        # DIFFUSION CONDITIONING, A20 Line 1
        # A21 
        s, z= self.diffusion_conditioning(
            asym_id = batch["asym_id"],
            residue_index = batch["residue_index"],
            entity_id = batch["entity_id"],
            token_index = batch["token_index"],
            sym_id = batch["sym_id"],
            t = t,
            s_trunk = s_trunk,
            s_inputs = s_inputs,
            z_trunk = z_trunk,
            inplace_safe=inplace_safe,
        )
        
        # ATOM ATTENTION ENCODER, A20 Line 3
        # A5
        a, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            ref_pos = batch["ref_pos"],
            ref_charge = batch["ref_charge"],
            ref_mask =  ref_mask.to(r_noisy.dtype),
            ref_element = batch["ref_element"],
            ref_atom_name_chars =  batch["ref_atom_name_chars"],
            ref_space_uid =  batch["ref_space_uid"],
            atom_mask = atom_mask.to(r_noisy.dtype),
            s_trunk = s_trunk,
            z = z,
            r = r_noisy,
            molecule_atom_lens = molecule_atom_lens,
            chunk_size = self.globals.chunk_size,
            use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
            inplace_safe=inplace_safe,
        )
        
        # Broadcast manually if batch size is larger than 1
        z = einops.repeat(z, "b ... -> (b n) ..." ,n = s.shape[0] // z.shape[0]) if (z.shape[0] != 1 and s.shape[0] != z.shape[0]) else z
        # DIFFUSION TRANSFORMER (TOKEN)
        # A20 Line 4
        a = add(a, self.proj_s(s), inplace=False)
        # A20 Line 5
        a = self.diffusion_transformer(
            a = a,
            s = s,
            z = z,
            mask = token_mask.to(dtype=z.dtype),
            chunk_size = self.globals.chunk_size,
            use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
            inplace_safe = inplace_safe,
        )
        # A20 Line 6
        a = self.layer_norm_a(a)
        # ATOM ATTENTION DECODER, A20 Line 7
        # A6
        r_update = self.atom_attention_decoder(
            a = a,
            q_skip = q_skip,
            c_skip = c_skip,
            p_skip = p_skip,
            atom_mask = atom_mask.to(r_noisy.dtype),
            molecule_atom_lens = einops.repeat(molecule_atom_lens, 'b ... -> (b n) ...', n = r_noisy.shape[0] // molecule_atom_lens.shape[0]) if (r_noisy.shape[0] != molecule_atom_lens.shape[0]) else molecule_atom_lens,
            chunk_size = self.globals.chunk_size,
            use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
            inplace_safe=inplace_safe,
        )
        return r_update