# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

from functools import partialmethod
from typing import Optional
from abc import ABC, abstractmethod
import einops
import torch
import torch.nn as nn

from aurobind.openfold.model.primitives import Linear, LayerNorm
from aurobind.openfold.utils.precision_utils import is_fp16_enabled
from aurobind.openfold.utils.tensor_utils import permute_final_dims


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Implements Algorithms 11 and 12.
    """
    @abstractmethod
    def __init__(self, c_z, c_hidden, _outgoing):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(BaseTriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = Linear(self.c_z, self.c_z, bias=False)
        self.linear_z = Linear(self.c_hidden, self.c_z, bias=False)

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # Incoming: equation='kjc,kic->ijc', but after permute_final_dims, 'ckj,cki->cij'
        # Outgoing: equation='ikc,jkc->ijc', but after permute_final_dims, 'cik,cjk->cij'

        if(_inplace_chunk_size is not None):
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i: i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i: i + _inplace_chunk_size, :, :]
                if self._outgoing:
                    a[..., i: i + _inplace_chunk_size, :, :] = torch.einsum('...cik,...cjk->...cij', a_chunk, b_chunk)
                else:
                    a[..., i: i + _inplace_chunk_size, :, :] = torch.einsum('...ckj,...cki->...cij', a_chunk, b_chunk)
            p = a
        else:
            if self._outgoing:
                p = torch.einsum('...cik,...cjk->...cij', a, b)
            else:
                p = torch.einsum('...ckj,...cki->...cij', a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        pass


class FusedTriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(FusedTriangleMultiplicativeUpdate, self).__init__(c_z=c_z,
                                                                c_hidden=c_hidden,
                                                                _outgoing=_outgoing)

        self.linear_ab_p = Linear(self.c_z, self.c_hidden * 2,bias=False)
        self.linear_ab_g = Linear(self.c_z, self.c_hidden * 2,bias=False)

    def forward(self,
                z: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                inplace_safe: bool = False,
                _add_with_inplace: bool = False,
                _inplace_chunk_size: Optional[int] = 256
                ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        
        ab = self.linear_ab_p(z)
        ab = ab * mask

        gate = self.linear_ab_g(z)
        ab = ab * self.sigmoid(gate)
        ab = permute_final_dims(ab, (2, 0, 1))
        
        ab = einops.rearrange(ab, 'b (c n) n1 n2 -> b c n n1 n2', n = 2)
        a, b = torch.split(ab, 1, dim = -3)
        a, b = a.squeeze(dim = -3), b.squeeze(dim = -3)
        
        a_std = a.std()
        b_std = b.std()
        if(is_fp16_enabled() and a_std != 0. and b_std != 0.):
            a = a / a.std()
            b = b / b.std()

        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(FusedTriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(FusedTriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(FusedTriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(FusedTriangleMultiplicativeUpdate.__init__, _outgoing=False)

