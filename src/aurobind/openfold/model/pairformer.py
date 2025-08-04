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
import torch.nn.functional as F
import numpy as np
import einops
from typing import Tuple, Sequence, Optional, List
from functools import partial
from aurobind.openfold.model.primitives import Linear, LayerNorm, softmax_no_cast
from aurobind.openfold.model.dropout import DropoutRowwise
from aurobind.openfold.model.outer_product_mean import OuterProductMean
from aurobind.openfold.model.triangular_attention import (
    TriangleAttention,
)
from aurobind.openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from aurobind.openfold.utils.chunk_utils import chunk_layer, ChunkSizeTuner
from aurobind.openfold.utils.tensor_utils import add,permute_final_dims,flatten_final_dims
from aurobind.openfold.model.primitives import Attention,_deepspeed_evo_attn
from aurobind.openfold.utils.atom_token_conversion import pad_at_dim,concat_previous_and_later_windows,slice_at_dim


class AF3Attention(Attention):
    """
    Standard multi-head attention using AlphaFold's default layer initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        conditioned: bool=False
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
            conditioned:
                Whether the attention should be conditioned on query data.
        """
        super(AF3Attention, self).__init__(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=gating,
        )

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=True 
        )
        
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        # without bias
        if conditioned:
            self.linear_o = Linear(
                self.c_hidden * self.no_heads,  self.c_q, bias = False
            )
        else:
            self.linear_o = Linear(
                self.c_hidden * self.no_heads,  self.c_q, bias = False
            )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads,bias=False
            )

class AdaLN(nn.Module):
    """
    The adaptive layer norm
    
    Implements the Algorithm 26
    """

    def __init__(
        self,
        c_a,
        c_s
    ):
        super().__init__()
        self.layer_norm_a = nn.LayerNorm(c_a, elementwise_affine = False, bias=False)
        self.layer_norm_s = nn.LayerNorm(c_s, elementwise_affine = True,  bias=False)

        self.linear_s_gamma = Linear(c_s, c_a, bias=True)
        
        self.sigmoid = nn.Sigmoid()

        self.linear_s_beta = Linear(c_s, c_a , bias=False)

    def forward(
        self,
        a,
        s
    ):
        a = self.layer_norm_a(a)
        s = self.layer_norm_s(s)
        a = a * self.sigmoid(self.linear_s_gamma(s)) + self.linear_s_beta(s)
        return a
    
class BiasAttention(nn.Module):
    """
    Simplified multi-head attention using AlphaFold's PairWeightAveraging layer initialization. Allows multiple bias vectors. It use the bias as the attention affinities.
    
    Implements Algorithm 10 (part of)
    """
    def __init__(
        self,
        c_v: int,
        c_hidden: int,
        no_heads: int,
    ):
        """
        Args:
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
        """
        super(BiasAttention, self).__init__()


        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads

        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_v, bias=False
        )

        self.linear_g = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def _prep_v(self,
        v_x: torch.Tensor,
    ):
        # [*, V, H * C_hidden]
        v = self.linear_v(v_x)

        # [*, V, H, C_hidden]

        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        
        # [*, H, V, C_hidden]
        v = v.transpose(-2, -3)

        return v

    def _wrap_up(self,
        o: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        g = self.sigmoid(self.linear_g(x))
    
        # [*, Q, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))
        o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def _attention(
        self,
        v: torch.Tensor,
        biases: List[torch.Tensor],
    ) -> torch.Tensor:
        a = sum(bias for bias in biases)
        a = softmax_no_cast(a, -1)
        o = torch.matmul(a, v)
        return o
    
    def forward(
        self,
        x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_deepspeed_evo_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, Q, C_v]
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed's EvoAttention for the attention
                


        Returns
            [*, Q, C_v] attention update
        """
        
        if biases is None:
            raise ValueError("Biases must be provided")

        v = self._prep_v(x)
        
        use_deepspeed_evo_attention = use_deepspeed_evo_attention and x.shape[-2] > 16
        if use_deepspeed_evo_attention:
            if len(biases) > 2:
                raise ValueError(
                    "If use_deepspeed_evo_attention is True, you may only "
                    "provide up to two bias terms"
                )
            o = _deepspeed_evo_attn(torch.zeros_like(v), torch.zeros_like(v), v, biases)
        else:
            o = self._attention(v, biases)
            o = o.transpose(-2, -3)
            
        o = self._wrap_up(o, x)
                
        return o


class Transition(nn.Module):
    """
    Feedforward transition layer with layer normalization and swish activation.
    
    Implements Algorithm 11
    """
    def __init__(self, c, transition_n):
        """
        Args:
            c:
                channel dimension
            transition_n:
                Factor by which c is multiplied to obtain hidden channel
                dimension
        """
        super(Transition, self).__init__()

        self.c = c
        self.transition_n = transition_n

        self.layer_norm = LayerNorm(self.c)

        self.linear = Linear(self.c, self.transition_n * self.c * 2, bias=False)
        self.swish = nn.SiLU()

        self.linear_o = Linear(self.transition_n * self.c, c, bias=False)

    def _transition(self, x, mask):

        x = self.layer_norm(x)
        a, b = torch.chunk(self.linear(x),chunks=2,dim = -1)
        x = self.swish(a) * b
        x = self.linear_o(x) 
        
        x = x * mask

        return x

    @torch.jit.ignore
    def _chunk(self,
        x: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"x": x, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
        )

    def forward(self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_token, C]  embedding
            mask:
                [*, N_token]  mask for the embedding
            chunk_size:
                If not None, the size of the chunks to use for the
        Returns:
            [*, N_token, C]  embedding update
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        # [*, N_token, 1]
        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            x = self._chunk(x, mask, chunk_size)
        else:
            x = self._transition(x=x, mask=mask)

        return x
    


class AttentionPairBias(nn.Module):
    """
    Attention layer with pair bias. (use in Pairformer)
    
    Implements Algorithm 24
    """
    def __init__(
        self, c_a,c_s,c_z, c_hidden, no_heads,window_size_row=None, window_size_col=None, conditioned=False,use_layer_norm_z=True,inf=1e9,eps=1e-8
    ):
        """
        Args:
            c_a:
                Input channel dimension
            c_s:
                Condition channel dimension
            c_z:
                Pair channel dimension
            c_hidden:
                Perhead hidden channel dimension
            no_heads:
                Number of attention heads
            window_size_row:
                Window size for local attention
            window_size_col:
                Window size for local attention
            conditioned:
                Whether the attention should be conditioned
        """
        super(AttentionPairBias, self).__init__()

        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.use_layer_norm_z = use_layer_norm_z
        self.conditioned = conditioned
        self.inf = inf
        self.window_size_row = window_size_row
        self.window_size_col = window_size_col
        if window_size_row is not None:
            if window_size_col is None:
                raise ValueError("Window size must be provided for both dimensions")
            
        if window_size_row is not None:
            if not conditioned:
                raise ValueError("Conditioned must be True for local attention")
        if conditioned:
            if window_size_row is not None:
                self.adaptive_layer_norm_q = AdaLN(self.c_a, self.c_s)
                self.adaptive_layer_norm_k = AdaLN(self.c_a, self.c_s)
            else:
                self.adaptive_layer_norm = AdaLN(self.c_a, self.c_s)
            self.linear_s  = Linear(self.c_s, self.c_a, bias=True)
            self.sigmoid = nn.Sigmoid()
        else:
            self.layer_norm = LayerNorm(self.c_a)

        if use_layer_norm_z:
            self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(c_z, self.no_heads, bias=False)
    
        self.mha = AF3Attention(
            self.c_a, self.c_a, self.c_a, self.c_hidden, self.no_heads,gating=True, conditioned=conditioned
        )

    @torch.jit.ignore
    def _chunk(self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": q_x,
            "kv_x": kv_x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha, 
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(q_x.shape[:-2]),
            _out=q_x if inplace_safe else None,
        )
        
    def _prep_inputs(self,
        x: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
        manual_broadcast: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        _, n_res = x.shape[-3:-1]
        if mask is None:
            mask = x.new_ones(
                x.shape[:-3] + (1, n_res),
            )

        # [*, 1, 1, 1, N_token]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        chunks = []

        for i in range(0, z.shape[-3], 256):
            z_chunk = z[..., i: i + 256, :, :]

            # [*, N_token, N_token, C_z]
            if self.use_layer_norm_z:
                z_chunk = self.layer_norm_z(z_chunk)
        
            # [*, N_token, N_token, no_heads]
            z_chunk = self.linear_z(z_chunk)

            chunks.append(z_chunk)
        
        z = torch.cat(chunks, dim=-3)
        # z is of [*, N_token, N_token, no_heads]
        # [*, 1, no_heads, N_token, N_token]
        z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)
        
        if manual_broadcast:
            if z.shape[0] != x.shape[0]:
                z = einops.repeat(z, 'b ... -> (b n) ...', n = x.shape[0] // z.shape[0])
            if mask_bias.shape[0] != x.shape[0]:
                mask_bias = einops.repeat(mask_bias, 'b ... -> (b n) ...', n = x.shape[0] // mask_bias.shape[0])
        
        return x, mask_bias, z
    
    def _prep_inputs_local(self,
        x: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        _, n_res = x.shape[-3:-1]
        
        # [*, N_window, 1, 1, window_size_col]
        mask_bias = (self.inf * (mask - 1))[..., :, None ,None , :]

        chunks = []

        for i in range(0, z.shape[-3], 256):
            z_chunk = z[..., i: i + 256, :, :]

            # [*,N_window, window_size_row, window_size_col, C_z]
            if self.use_layer_norm_z:
                z_chunk = self.layer_norm_z(z_chunk)
        
            # [*,N_window, window_size_row, window_size_col, no_heads]
            z_chunk = self.linear_z(z_chunk)

            chunks.append(z_chunk)
        
        z = torch.cat(chunks, dim=-3)
        # z is of [*,N_window, window_size_row, window_size_col, no_heads]
        # [*, N_window, no_heads, window_size_row, window_size_col]
        z = permute_final_dims(z, (2, 0, 1))
        
        return x, mask_bias, z
    
    def forward_local(
        self, 
        a: torch.Tensor,
        s: torch.Tensor, 
        z: torch.Tensor,
        single_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, I, C_a] input tensor (e.g. the single representation)
            s:
                [*, I, C_s] input tensor (e.g. the single condition)
            z:
                [*, W, I, J, C_z] input tensor (e.g. the windowed pair representation)
        Returns:
            [*, I, C_a] output tensor
        """ 

        if single_mask is None:
            # [*, I]
            single_mask = a.new_ones(
                a.shape[:-1],
        )
    
        if not self.conditioned:
            raise ValueError("Conditioned must be True for local attention")


        a_row = self.adaptive_layer_norm_q(a, s)
        a_col = self.adaptive_layer_norm_k(a, s)

        window_size_row, window_size_col, seq_len = self.window_size_row, self.window_size_col, a.shape[-2],
        if window_size_row != z.shape[-3] or  window_size_col != z.shape[-2]:
            raise ValueError("Window size must match the windowed pair representation")
        
        
        padding_needed_row = (window_size_row - (seq_len % window_size_row)) % window_size_row
        padding_needed_col = (window_size_col - (seq_len % window_size_col)) % window_size_col

        if padding_needed_row > 0:
            a_row = pad_at_dim(a_row, (0, padding_needed_row), value = 0., dim = -2)
            single_mask_row = pad_at_dim(single_mask, (0, padding_needed_row), value = 0., dim = -1)
            
            a_col = pad_at_dim(a_col, (0, padding_needed_row), value = 0., dim = -2)
            single_mask_col = pad_at_dim(single_mask, (0, padding_needed_row), value = 0., dim = -1)
        else:
            single_mask_row = single_mask
            single_mask_col = single_mask

        q_a = einops.rearrange(a_row, 'b (n w) d -> b n w d', w = window_size_row)
        kv_a = einops.rearrange(a_col, 'b (n w) d -> b n w d', w = window_size_col // 8)
        kv_a = concat_previous_and_later_windows(kv_a, dim_seq = -3, dim_window = -2)
        single_mask_row = einops.rearrange(single_mask_row, 'b (n w) -> b n w', w = window_size_row)
        single_mask_col = einops.rearrange(single_mask_col, 'b (n w) -> b n w', w = window_size_col // 8)
        single_mask_col = concat_previous_and_later_windows(single_mask_col, dim_seq = -2, dim_window = -1)
        
        q_a, mask_bias, z = self._prep_inputs_local(
            q_a, z, single_mask_col, inplace_safe=inplace_safe
        )
        
        biases = [mask_bias]
        
        if(z is not None):
            biases.append(z)

        if chunk_size is not None:
            a = self._chunk(
                q_x =q_a,
                kv_x =kv_a,
                biases=biases, 
                chunk_size=chunk_size, 
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
            )
        else:
            
            a = self.mha(
                q_x=q_a, 
                kv_x=kv_a, 
                biases=biases, 
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            )
        a = a * single_mask_row.unsqueeze(-1)
        a = einops.rearrange(a, "b n w d -> b (n w) d")
        a = slice_at_dim(a, slice(0, seq_len), dim = -2)

        a = a * self.sigmoid(self.linear_s(s))

        return a
        
    
    def forward(
        self, 
        a: torch.Tensor,
        s: torch.Tensor, 
        z: torch.Tensor,
        single_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, I, C_a] input tensor (e.g. the single representation)
            s:
                [*, I, C_s] input tensor (e.g. the single condition)
            z:
                [*, I, J, C_z] input tensor (e.g. the pair representation)
        Returns:
            [*, I, C_a] output tensor
        """ 
        use_deepspeed_evo_attention = False
        if self.window_size_row is not None:
            return self.forward_local(
                a, s, z, single_mask, pair_mask, chunk_size, use_deepspeed_evo_attention=False, inplace_safe=inplace_safe
            )
        if single_mask is None:
            # [*, I]
            single_mask = a.new_ones(
                a.shape[:-1],
            )
        if self.conditioned:
            a = self.adaptive_layer_norm(a, s)
        else:
            a = self.layer_norm(a)

        a = a.unsqueeze(-3)
        single_mask = single_mask.unsqueeze(-2)
        
        a, mask_bias, z = self._prep_inputs(
            a, z, single_mask, inplace_safe=inplace_safe, manual_broadcast=use_deepspeed_evo_attention
        )

            
        biases = [mask_bias]
        
        if(z is not None):
            biases.append(z)
        
        if chunk_size is not None:
            a = self._chunk(
                q_x=a,
                kv_x=a, 
                biases=biases, 
                chunk_size=chunk_size, 
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
            )
        else:
            a = self.mha(
                q_x=a, 
                kv_x=a, 
                biases=biases, 
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            )
        a = a * single_mask.unsqueeze(-1)
        a = a.squeeze(-3)
        
        if self.conditioned:

            a = a * self.sigmoid(self.linear_s(s))
            
        return a



class PairStack(nn.Module):
    """
    The pairstack use in Pairformer, MSAModule, TemplateEmbedder
    
    Implements Algorithm 8 (part of), 16 (part of), 17 (part of)
    """
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float
    ):
        super(PairStack, self).__init__()


        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            starting=True,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            starting=False,
            inf=inf,
        )

        self.pair_transition = Transition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_token, N_token, C_z] pair embedding
            pair_mask:
                [*, N_token, N_token] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
            inplace_safe:
                Whether to perform in-place operations
            _mask_trans:
                Whether to apply the mask to the transition layer
            _attn_chunk_size:
                Inference-time subbatch size for attention. Acts as a minimum if 
                self.tune_chunk_size is True
        Returns:
            [*, N_token, N_token, C_z] pair embedding
        """
        pair_trans_mask = pair_mask if _mask_trans else None

        if (_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )

        z = z + self.ps_dropout_row_layer(tmu_update)


        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )

        z = z + self.ps_dropout_row_layer(tmu_update)


        del tmu_update

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )
        
        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_end(
                        z,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = add(z,
                self.pair_transition(
                    z, mask=pair_trans_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
        )

        return z
    
    
class PairformerBlock(nn.Module):
    """
    One Block of the Pairformer model.
    
    Implements Algorithm 17 (part of)
    """
    def __init__(self,
        c_s: int,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_single: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
    ):
        """
        Args:
            c_s:
                Single channel dimension
            c_z:
                Pair channel dimension
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            no_heads_single:
                Number of heads used for single attention
            no_heads_pair:
                Number of heads used for pair attention
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            pair_dropout:
                Dropout used for pair activations
            inf:
                A large number to be used in computing the attention mask
            eps:
                A small number to be used in computing the attention mask
        
        """
        super(PairformerBlock, self).__init__()

        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps
        )

        
        self.attention_pair_bias = AttentionPairBias(
            c_a=c_s,
            c_s=None,
            c_z=c_z,
            c_hidden=c_s // no_heads_single,
            no_heads = no_heads_single,
            window_size_row=None,
            window_size_col=None,
            conditioned=False,
        )
        self.single_transition = Transition(
            c_s,
            transition_n,
        )

    
    def forward(self,
        s: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, N_token, C_s] single embedding
            z:
                [*, N_token, N_token, C_z] pair embedding
            single_mask:
                [*, N_token] single mask
            pair_mask:
                [*, N_token, N_token] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
            inplace_safe:
                Whether to perform in-place operations
            _mask_trans:
                Whether to apply the mask to the transition layer
            _attn_chunk_size:
                Inference-time subbatch size for attention. Acts as a minimum if 
                self.tune_chunk_size is True
        """

        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        input_tensors = [s, z]
        
        s, z = input_tensors

            
        if (not inplace_safe):
            input_tensors = [s, z]
            
        del s , z

        z = self.pair_stack(
            z=input_tensors[1], 
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size
        )
        s = input_tensors[0]
            
        s = add(s,
                self.attention_pair_bias(
                    a=s,
                    s=None,
                    z=z,
                    single_mask=single_mask,
                    pair_mask=pair_mask,
                    chunk_size=_attn_chunk_size,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    inplace_safe=inplace_safe,
                ),
                inplace=inplace_safe,
        )
        
        single_trans_mask = single_mask if _mask_trans else None
        s = add(s,
                self.single_transition(
                    s, mask=single_trans_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
        )
        
        return s, z

class PairformerStack(nn.Module):
    """
    Main Pairformer trunk.

    Implements Algorithm 17
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_heads_single: int,
        no_blocks: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single channel dimension
            c_z:
                Pair channel dimension
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            no_heads_pair:
                Number of heads used for pair attention
            no_heads_single:
                Number of heads used for attention pair bias
            no_blocks:
                Number of Pairformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            pair_dropout:
                Dropout used for pair activations
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super(PairformerStack, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = PairformerBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                no_heads_single=no_heads_single,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)


        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if(tune_chunk_size):
            self.chunk_size_tuner = ChunkSizeTuner()

    def forward(self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, , N_token, C_m] single embedding
            z:
                [*, N_token, N_token, C_z] pair embedding
            single_mask:
                [*, N_token] single mask
            pair_mask:
                [*, N_token, N_token] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
            inplace_safe:
                Whether to perform in-place operations
            _mask_trans:
                Whether to apply the mask to the transition layer
        Returns:
            z:
                [*, N_token, N_token, C_z] pair embedding
            s:
                [*, N_token, C_s] single embedding
        """ 
        
        for block in self.blocks:
            s, z = block(s=s,
                        z=z,
                        single_mask=single_mask,
                        pair_mask=pair_mask,
                        chunk_size=chunk_size,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        inplace_safe=inplace_safe,
                        _mask_trans=_mask_trans)
    
        return s, z

class MSAPairWeightedAveraging(nn.Module):
    """
    Multi-head self-attention with pair-weighted averaging.
    
    Implements Algorithm 10
    """
    def __init__(
        self,
        c_m,
        c_hidden,
        no_heads,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_m:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAPairWeightedAveraging, self).__init__()

        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_m)

        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(
            self.c_z, self.no_heads, bias=False
        )
        
        self.mha = BiasAttention(
            self.c_m,  
            self.c_hidden, 
            self.no_heads,
        )

    @torch.jit.ignore
    def _chunk(self, 
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
    ) -> torch.Tensor:
        def fn(m, biases):
            m = self.layer_norm_m(m)
            return self.mha(
                x=m, 
                biases=biases,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            )

        inputs = {"m": m}
        if(biases is not None):
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
        use_deepspeed_evo_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_msa, N_token]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )

        if not use_deepspeed_evo_attention:
            mask = torch.max(mask, dim = -2, keepdim = True).values
        
        # [*, 1, 1, 1, N_token]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        chunks = []

        for i in range(0, z.shape[-3], 256):
            z_chunk = z[..., i: i + 256, :, :]

            # [*, N_token, N_token, C_z]
            z_chunk = self.layer_norm_z(z_chunk)
        
            # [*, N_token, N_token, no_heads]
            z_chunk = self.linear_z(z_chunk)

            chunks.append(z_chunk)
        
        z = torch.cat(chunks, dim=-3)
        
        # [*, 1, no_heads, N_token, N_token]
        z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        use_deepspeed_evo_attention: bool = False,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,

    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_msa, N_token, C_m] MSA embedding
            z:
                [*, N_token, N_token, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_msa, N_token] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
                
        """
        
        m, mask_bias, z = self._prep_inputs(
            m, z, mask, inplace_safe=inplace_safe,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )

        biases = [mask_bias]
        if(z is not None):
            biases.append(z)
        
        if chunk_size is not None:  
            m = self._chunk(
                m, 
                biases, 
                chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            )
        else:
            m = self.layer_norm_m(m)
            m = self.mha(
                x=m, 
                biases=biases,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            )

        return m



class MSABlock(nn.Module):
    """ 
    A block of thee MSAModule
    
    Implements Part of Algorithm 8
    """
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
    ):
        super(MSABlock, self).__init__()

        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(    
                    c_m=c_m,
                    c_z=c_z,
                    c_hidden=c_hidden_msa_att,
                    no_heads=no_heads_msa,
                    inf=inf,
                )
            
        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.msa_transition = Transition(
            c_m,
            transition_n=transition_n,
        )

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps
        )
        
        
    def _compute_opm(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m, z = input_tensors


        opm = self.outer_product_mean(
            m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
        )

        z = add(z, opm, inplace=inplace_safe)
        del opm

        return m, z

    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_msa, N_token, C_m] MSA embedding
            z:
                [*, N_token, N_token, C_z] pair embedding
            msa_mask:
                [*, N_msa, N_token] MSA mask
            pair_mask:
                [*, N_token, N_token] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
            inplace_safe:
                Whether to perform in-place operations
            _mask_trans:
                Whether to apply the mask to the transition layer
            _attn_chunk_size:
                Inference-time subbatch size for attention. Acts as a minimum if 
                self.tune_chunk_size is True
        """

        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size
       
        input_tensors = [m, z]

        m, z = self._compute_opm(input_tensors=input_tensors,
                                    msa_mask=msa_mask,
                                    chunk_size=chunk_size,
                                    inplace_safe=inplace_safe,)

        m = add(m,
            self.msa_dropout_layer(
                self.msa_pair_weighted_averaging(
                    m.clone() if torch.is_grad_enabled() else m, 
                    z=z.clone() if torch.is_grad_enabled() else z, 
                    mask=msa_mask, 
                    chunk_size=_attn_chunk_size,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                )
            ),
            inplace=inplace_safe,
        )

        if (not inplace_safe):
            input_tensors = [m, z]

        del m, z

        def fn(input_tensors):
            m, z = input_tensors

            m = add(
                m,
                self.msa_transition(
                    m, mask=msa_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
            )

            if (not inplace_safe):
                input_tensors = [m, z]

            del m, z

            z = self.pair_stack(
                input_tensors[1],
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
                _attn_chunk_size=_attn_chunk_size,
            )

            m = input_tensors[0]

            return m, z

        m, z = fn(input_tensors)

        return m, z
    
class MSAModuleStack(nn.Module):
    """
    The Stack of MSAModule (Embedder implement in the embedders.py)
    
    Implements Algorithm 8 (part of)
    """
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        super(MSAModuleStack, self).__init__()
 
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = MSABlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)
            
        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if(tune_chunk_size):
            self.chunk_size_tuner = ChunkSizeTuner()
            
    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_token, C_m] extra MSA embedding
            z:
                [*, N_token, N_token, C_z] pair embedding
            chunk_size: Inference-time subbatch size for Pairformer modules
            use_deepspeed_evo_attention: Whether to use DeepSpeed memory-efficient kernel
            msa_mask:
                Optional [*, N_extra, N_token] MSA mask
            pair_mask:
                Optional [*, N_token, N_token] pair mask
            inplace_safe:
                Whether to perform in-place operations
            _mask_trans:
                Whether to apply the mask to the transition layer
        Returns:
            [*, N_token, N_token, C_z] pair update
        """
        for block in self.blocks:
            m, z = block(
                m=m,
                z=z,
                msa_mask=msa_mask, 
                pair_mask=pair_mask, 
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )

        return m, z
