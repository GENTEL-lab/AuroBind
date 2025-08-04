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
import einops
from typing import Tuple
import torch.nn.functional as F


def aggregate_fn(original_seqs, attention_mask):
    """
    Aggregates the original_seqs by only reshaping the tensor.
    
    Args:
        original_seqs (list of torch.torch.Tensor): List of original_seqs to aggregate. Shape: [batch_size, num_tokens, 24,...]
        attention_mask (torch.torch.Tensor): Attention mask to aggregate. Shape: [batch_size, seq_len]

    Returns:
        aggregated_seqs (list of torch.torch.Tensor): List of aggregated original_seqs. Shape: [batch_size, num_atoms, ...]
        , where num_atoms = num_tokens * 24
    """
    batch_size, num_tokens, max_num_atoms_per_token = attention_mask.shape[:3]
    if max_num_atoms_per_token != 24:
        raise ValueError("Only 24 atoms per token is supported")
    
    aggregated_seqs = []
    for t in original_seqs:
        aggregated_seqs.append(einops.rearrange(t, 'b n a ... -> b (n a) ...', a = 24))
    
    def reverse_fn(aggregated_seqs):
        original_seqs = []
        for t in aggregated_seqs:
            original_seqs.append(einops.rearrange(t, 'b (n a) ... -> b n a ...', a = 24))
        return original_seqs

    return aggregated_seqs, reverse_fn

def aggregate_fn_advanced(original_seqs, attention_mask):
    """
    Aggregates the original_seqs and attention mask by removing the padding tokens.

    Args:
        original_seqs (list of torch.torch.Tensor): List of original_seqs to aggregate. Shape: [batch_size, num_tokens, 24,...]
        attention_mask (torch.torch.Tensor): Attention mask to aggregate. Shape: [batch_size, seq_len]

    Returns:
        aggregated_seqs (list of torch.torch.Tensor): List of aggregated original_seqs. Shape: [batch_size, num_atoms, ...]
        reverse_fn (function): Function to reverse the aggregation
    """
    # Validate batch size
    batch_size, num_tokens, max_num_atoms_per_token = attention_mask.shape[:3]

    if max_num_atoms_per_token != 24:
        raise ValueError("Only 24 atoms per token is supported")
    
    # Ensure attention_mask is a boolean tensor
    attention_mask = attention_mask.bool()
    
    # Extract valid sequence lengths
    num_atoms = attention_mask.sum(dim=(1, 2))
    
    if (num_atoms == 0).any():
        raise ValueError("Some sequences have zero atoms. Please remove them before aggregation.")
    
    max_atoms = num_atoms.max().view(())
    
    # Create Output attention mask
    range_tensor = torch.arange(max_atoms, device=attention_mask.device)
    range_tensor = einops.repeat(range_tensor, 'm -> b m', b=batch_size) 
    output_attention_mask = range_tensor < num_atoms.unsqueeze(1)
    
    # Create Empty output tensor
    aggregated_seqs = []
    for t in original_seqs:
        aggregated_seqs.append(torch.zeros([batch_size, max_atoms, *t.shape[3:]], dtype=t.dtype, device=t.device))
    
    # Aggregate original_seqs based on the attention mask
    for i in range(len(original_seqs)):
        aggregated_seqs[i][output_attention_mask] = original_seqs[i][attention_mask]

    # Define reverse function
    def reverse_fn(aggregated_seqs):
        """
        Reconstructs the original tensor from the aggregated tensor.
        """
        original_seqs = []
        
        for aggregated_seq in aggregated_seqs:
            new_batch_size = aggregated_seq.shape[0]
            
            if new_batch_size != batch_size:
                # it is just repeating the batch_size times
                original_seq = torch.zeros([new_batch_size, num_tokens, max_num_atoms_per_token, *aggregated_seq.shape[2:]], dtype=aggregated_seq.dtype, device=aggregated_seq.device)
                # repeat the attention mask
                new_attention_mask = einops.repeat(attention_mask, 'b ... -> (b m) ...', m=new_batch_size // batch_size)
                new_output_attention_mask = einops.repeat(output_attention_mask, 'b ... -> (b m) ...', m=new_batch_size // batch_size)
                original_seq[new_attention_mask] = aggregated_seq[new_output_attention_mask]

            else:
                original_seq = torch.zeros([batch_size, num_tokens,max_num_atoms_per_token, *aggregated_seq.shape[2:]], dtype=aggregated_seq.dtype, device=aggregated_seq.device)
                original_seq[attention_mask] = aggregated_seq[output_attention_mask]

            original_seqs.append(original_seq)
        return original_seqs

    return aggregated_seqs, reverse_fn



def slice_at_dim(
    t: torch.Tensor,
    dim_slice: slice,
    *,
    dim: int
):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def concat_previous_and_later_windows(
    t: torch.Tensor,
    *,
    dim_seq: int,
    dim_window: int,
    
):
    assert dim_seq == dim_window - 1, 'dim_seq should be dim_window - 1'
    # kind of hard coded for now: 3 , 2 , 3
    first = slice_at_dim(t, slice(None, 8), dim = dim_seq)
    first = torch.flatten(first, start_dim = dim_seq, end_dim =dim_window)
    first = first.unsqueeze(dim_seq)
    last = slice_at_dim(t, slice(-8, None), dim = dim_seq)
    last = torch.flatten(last, start_dim = dim_seq, end_dim =dim_window)
    last = last.unsqueeze(dim_seq)
    
    t = pad_at_dim(t, (3, 4), dim = dim_seq, value = 0.)
    
    left = torch.cat((
        slice_at_dim(t, slice(None, -7, 2), dim = dim_seq),
        slice_at_dim(t, slice(1, -6 , 2), dim = dim_seq),
        slice_at_dim(t, slice(2, -5, 2), dim = dim_seq),
        ), dim = dim_window)
    
    middle = torch.cat((
        slice_at_dim(t, slice(3, -4, 2), dim = dim_seq),
        slice_at_dim(t, slice(4, -3, 2),  dim = dim_seq),
        ), dim = dim_window)
    
    right = torch.cat((
        slice_at_dim(t, slice(5, -2, 2), dim = dim_seq),
        slice_at_dim(t, slice(6, -1, 2), dim = dim_seq),
        slice_at_dim(t, slice(7, None, 2), dim = dim_seq),
        ), dim = dim_window)

    t = torch.cat((
        left,
        middle,
        right
    ), dim = dim_window)
        
    t = torch.cat((
        first,first,
        slice_at_dim(t, slice(2, -2), dim = dim_seq),
        last,last
    ), dim = dim_seq)
    
    return t.contiguous()

def lens_to_mask(lens, max_len):

    device = lens.device
    
    if max_len is None:
        max_len = lens.amax()
    arange = torch.arange(max_len, device = device)
    
    return torch.lt(arange.unsqueeze(0), lens.unsqueeze(-1))

def pad_to_multiple(
    t: torch.Tensor,
    multiple: int,
    *,
    dim = -1,
    value = 0.
):
    seq_len = t.shape[dim]
    padding_needed = (multiple - (seq_len % multiple)) % multiple

    if padding_needed == 0:
        return t

    return pad_at_dim(t, (0, padding_needed), dim = dim, value = value)

# A NATIVE REPEAT CONSECUTIVE WITH LENS FUNCTION
def repeat_consecutive_with_lens(
    feats,
    lens,
):

    # Repeat the features seq // 24 times
    feats = einops.repeat(feats, 'b n a ... -> b (n 24) a ...')
    
    return feats


def repeat_consecutive_with_lens_advanced(
    feats,
    lens,
):
    
    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens

    mask = lens_to_mask(lens, max_len=None)
    
    # derive arange

    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device = device)

    cumsum_len = lens.cumsum(dim = -1)
    offsets = F.pad(cumsum_len, (1, -1), value = 0)
    indices = einops.rearrange(arange, 'n -> 1 1 n') + offsets.unsqueeze(-1)
    
    total_lens = lens.sum(dim = -1)
    output_mask = lens_to_mask(total_lens, max_len=None)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, int(max_len + 1)), device = device, dtype = torch.long)

    indices = indices.masked_fill(~mask, max_len) 
    indices = einops.rearrange(indices, 'b n w -> b (n w)')

    # scatter
    seq_arange = torch.arange(seq, device = device)
    seq_arange = einops.repeat(seq_arange, 'n -> (n w)', w = window_size)

    output_indices = output_indices.scatter(1, indices, seq_arange.unsqueeze(0).expand_as(indices))

    # remove sink
    
    output_indices = output_indices[:, :-1]

    # gather
    output = torch.gather(feats, dim=1, index=output_indices.unsqueeze(-1).expand(-1, -1, *feats.shape[2:]))

    # final mask

    mask_value = False if dtype == torch.bool else 0

    output = torch.where(output_mask.unsqueeze(-1).expand_as(output), output, mask_value)

    
    return output

def pad_and_window(
    t,
    window_size: int
):
    t = pad_to_multiple(t, window_size, dim = 1)
    t = einops.rearrange(t, 'b (n w) ... -> b n w ...', w = window_size)
    return t


# A NATIVE MEAN POOL WITH LENS FUNCTION
def mean_pool_with_lens(feats, feats_mask, lens, eps = 1e-6):
    """
    Mean pools the features based on the lens.
    """
    seq_len = feats.shape[1]
    
    # 24 atoms per token
    if seq_len % 24 != 0:
        raise ValueError("Sequence length must be divisible by 24")
    
    # First rearrange the features to have a shape of [batch_size, num_tokens, 24, ...]
    feats = einops.rearrange(feats, 'b (n a) ... -> b n a ...', a = 24)
    feats_mask = einops.rearrange(feats_mask, 'b (n a) ... -> b n a ...', a = 24)
    
    # Do the mean pooling
    feats = torch.sum(feats * feats_mask.unsqueeze(-1), dim = 2) / (torch.sum(feats_mask, dim = 2, keepdim = True) + eps)
    
    return feats
    

def pad_at_dim(
    t,
    pad: Tuple[int, int],
    *,
    dim = -1,
    value = 0.
):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)