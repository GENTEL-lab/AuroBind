# Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro
#
# Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Optional
from collections.abc import Iterator, Mapping, Sequence
import itertools
import bisect

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from aurobind.data import const
from aurobind.data.feature.featurizer import BoltzFeaturizer
from aurobind.data.feature.pad import pad_to_max
from aurobind.data.tokenize.boltz import BoltzTokenizer
from aurobind.data.types import (
    MSA,
    Connection,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    Structure,
)


_BUCKETS = (
    256,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
)

def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(target_dir / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    residue_constraints = None
    if constraints_dir is not None:
        residue_constraints = ResidueConstraints.load(
            constraints_dir / f"{record.id}.npz"
        )

    return Input(structure, msas, residue_constraints=residue_constraints)

def calculate_bucket_size(
    num_tokens: int, buckets: Sequence[int] | None
) -> int:
  """Calculates the bucket size to pad the data to."""
  if buckets is None:
    return num_tokens

  if not buckets:
    raise ValueError('Buckets must be non-empty.')

  if not all(prev < curr for prev, curr in itertools.pairwise(buckets)):
    raise ValueError(
        f'Buckets must be in strictly increasing order. Got {buckets=}.'
    )

  bucket_idx = bisect.bisect_left(buckets, num_tokens)

  if bucket_idx == len(buckets):
    print(
        'Creating a new bucket of size %d since the input has more tokens than'
        ' the largest bucket size %d. This may trigger a re-compilation of the'
        ' model. Consider additional large bucket sizes to avoid excessive'
        ' re-compilation.',
        num_tokens,
        buckets[-1],
    )
    return num_tokens

  return buckets[bucket_idx]

def construct_empty_template_features(
    input_features, 
    device
    ):
    """
    Construct empty template features for the model.
    Parameters
    ----------
    input_features : dict
        The input features to construct the empty template features for.
    device : torch.device
        The device to move the template features to.
    Returns
    -------
    dict
        The constructed template features.
    """
    ## empty template feature
    template_aatype = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1]))
    template_aatype = F.one_hot(template_aatype.long(),num_classes=31).float()
    template_distogram = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1], input_features['aatype'].shape[1], 39))
    template_pseudo_beta_mask = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1]))
    template_unit_vector = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1], input_features['aatype'].shape[1], 3))
    template_backbone_frame_mask = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1]))
    
    template_features = {}
    template_features['template_aatype'] = template_aatype.float().to(device)
    template_features['template_distogram'] = template_distogram.float().to(device)
    template_features['template_pseudo_beta_mask'] = template_pseudo_beta_mask.float().to(device)
    template_features['template_unit_vector'] = template_unit_vector.float().to(device)
    template_features['template_backbone_frame_mask'] = template_backbone_frame_mask.float().to(device)

    return template_features

def transform(boltz_input_features):
    """
    Transform the input features to the format required by the model.
    Parameters
    ----------
    boltz_input_features : dict
        The input features to transform.
    Returns
    -------
    dict
        The transformed input features.
    """
    
    ### calculate the bucket size for padding
    total_tokens = len(boltz_input_features['token_index'][0])
    padded_token_length = calculate_bucket_size(
        total_tokens, _BUCKETS
    )
    
    boltz_input_features['token_index'] = F.pad(boltz_input_features['token_index']+1, (0, padded_token_length - boltz_input_features['token_index'].shape[1]), 'constant', 0)
    boltz_input_features['residue_index'] = F.pad(boltz_input_features['residue_index']+1, (0, padded_token_length - boltz_input_features['residue_index'].shape[1]), 'constant', 0)
    boltz_input_features['asym_id'] = F.pad(boltz_input_features['asym_id']+1, (0, padded_token_length - boltz_input_features['asym_id'].shape[1]), 'constant', 0)
    boltz_input_features['entity_id'] = F.pad(boltz_input_features['entity_id']+1, (0, padded_token_length - boltz_input_features['entity_id'].shape[1]), 'constant', 0)
    boltz_input_features['sym_id'] = F.pad(boltz_input_features['sym_id']+1, (0, padded_token_length - boltz_input_features['sym_id'].shape[1]), 'constant', 0)
    boltz_input_features['res_type'] = F.pad(boltz_input_features['res_type'], (0, 0, 0, padded_token_length - boltz_input_features['res_type'].shape[1]), 'constant', 0)
    boltz_input_features['token_pad_mask'] = F.pad(boltz_input_features['token_pad_mask'], (0, padded_token_length - boltz_input_features['token_pad_mask'].shape[1]), 'constant', 0)
    boltz_input_features['token_bonds'] = F.pad(boltz_input_features['token_bonds'], (0, 0, 0, padded_token_length - boltz_input_features['token_bonds'].shape[2], 0, padded_token_length - boltz_input_features['token_bonds'].shape[1]), 'constant', 0).squeeze(-1)
 
    ### msa feat
    boltz_input_features['msa'] = F.pad(boltz_input_features['msa'], (0, 0, 0, padded_token_length - boltz_input_features['msa'].shape[2], 0, 0), 'constant', 0)[..., :32]
    boltz_input_features['msa_mask'] = F.pad(boltz_input_features['msa_mask'], (0, padded_token_length - boltz_input_features['msa_mask'].shape[2]), 'constant', 0)
    boltz_input_features['deletion_value'] = F.pad(boltz_input_features['deletion_value'], (0, padded_token_length - boltz_input_features['deletion_value'].shape[2]), 'constant', 0)
    boltz_input_features['profile'] = F.pad(boltz_input_features['profile'], (0, 0, 0, padded_token_length - boltz_input_features['profile'].shape[1]), 'constant', 0)[..., :31]
    boltz_input_features['deletion_mean'] = F.pad(boltz_input_features['deletion_mean'], (0, padded_token_length - boltz_input_features['deletion_mean'].shape[1]), 'constant', 0)
    boltz_input_features['has_deletion'] = F.pad(boltz_input_features['has_deletion'], (0, padded_token_length - boltz_input_features['has_deletion'].shape[2]), 'constant', 0)
    
    input_features = {}
    
    input_features['token_index'] = boltz_input_features['token_index'].int()
    input_features['residue_index'] = boltz_input_features['residue_index'].int()
    input_features['asym_id'] = boltz_input_features['asym_id'].int()
    input_features['entity_id'] = boltz_input_features['entity_id'].int()
    input_features['sym_id'] = boltz_input_features['sym_id'].int()
    input_features['aatype'] = boltz_input_features['res_type'][..., 0:31].float()
    input_features['seq_mask'] = boltz_input_features['token_pad_mask'].bool()
    input_features['token_bonds'] = boltz_input_features['token_bonds']
    
    input_features['is_protein'] = F.pad((boltz_input_features['mol_type'] == const.chain_type_ids["PROTEIN"]), (0, padded_token_length - boltz_input_features['mol_type'].shape[1]), 'constant', False)
    input_features['is_rna'] = F.pad((boltz_input_features['mol_type'] == const.chain_type_ids["RNA"]), (0, padded_token_length - boltz_input_features['mol_type'].shape[1]), 'constant', False)
    input_features['is_dna'] = F.pad((boltz_input_features['mol_type'] == const.chain_type_ids["DNA"]), (0, padded_token_length - boltz_input_features['mol_type'].shape[1]), 'constant', False)
    input_features['is_ligand'] = F.pad((boltz_input_features['mol_type'] == const.chain_type_ids["NONPOLYMER"]), (0, padded_token_length - boltz_input_features['mol_type'].shape[1]), 'constant', False)
    
    
    input_features['msa'] = boltz_input_features['msa']
    input_features['msa_mask'] = boltz_input_features['msa_mask']
    input_features['deletion_value'] = boltz_input_features['deletion_value']
    input_features['deletion_mean'] = boltz_input_features['deletion_mean']
    input_features['profile'] = boltz_input_features['profile'][..., 0:31]
    input_features['num_alignments'] = boltz_input_features['msa_mask'].max(-1).values.sum(1).int()
    input_features['has_deletion'] = boltz_input_features['has_deletion'].float()
    
            
    ########## Atom level feat ##########
    
    input_features['atom_nums'] = boltz_input_features['atom_nums']
    input_features['center_idx'] = boltz_input_features['center_idx']
    
    atom_counts_per_token =  boltz_input_features['atom_to_token'].sum(dim=1)
    atom_pad_mask = boltz_input_features['atom_pad_mask']
    atom_to_token = boltz_input_features['atom_to_token'][atom_pad_mask==1].unsqueeze(0) 
    ref_pos = boltz_input_features['ref_pos'][atom_pad_mask==1].unsqueeze(0)              
    atom_counts_per_token = atom_to_token.sum(dim=1)      
    token_len = atom_counts_per_token.shape[1]             
    max_atoms_per_token = 24

    atom_token_indices = atom_to_token.argmax(dim=2)  

    pad_ref_pos = torch.zeros((1, token_len, max_atoms_per_token, 3), 
                        device=ref_pos.device, 
                        dtype=ref_pos.dtype)

    for token_idx in range(token_len):
        mask = (atom_token_indices[0] == token_idx)  
        current_atoms = ref_pos[0, mask]  
        num_atoms = current_atoms.shape[0]
        if num_atoms > 0:
            pad_ref_pos[0, token_idx, :num_atoms] = current_atoms
    pad_ref_pos = F.pad(pad_ref_pos, (0, 0, 0, 0, 0, padded_token_length - token_len), 'constant', 0)

    
    ref_element = boltz_input_features['ref_element'][atom_pad_mask==1].unsqueeze(0)  
    pad_ref_element = torch.zeros((1, token_len, max_atoms_per_token, 128),
                        device=ref_element.device, 
                        dtype=ref_element.dtype)
    for token_idx in range(token_len):
        mask = (atom_token_indices[0] == token_idx)  
        current_elements = ref_element[0, mask]  
        num_atoms = current_elements.shape[0]
        if num_atoms > 0:
            pad_ref_element[0, token_idx, :num_atoms] = current_elements
    pad_ref_element = F.pad(pad_ref_element, (0, 0, 0, 0, 0, padded_token_length - token_len), 'constant', 0)
    
    
    ref_charge = boltz_input_features['ref_charge'][atom_pad_mask==1].unsqueeze(0)  # [1, 1502]
    pad_ref_charge = torch.zeros((1, token_len, max_atoms_per_token), 
                        device=ref_charge.device, 
                        dtype=ref_charge.dtype)
    for token_idx in range(token_len):
        mask = (atom_token_indices[0] == token_idx) 
        current_charges = ref_charge[0, mask]  
        num_atoms = current_charges.shape[0]
        if num_atoms > 0:
            pad_ref_charge[0, token_idx, :num_atoms] = current_charges
    pad_ref_charge = F.pad(pad_ref_charge, (0, 0, 0, padded_token_length - token_len), 'constant', 0)


    ref_atom_name_chars= boltz_input_features['ref_atom_name_chars'][atom_pad_mask==1].unsqueeze(0)  
    pad_ref_atom_name_chars = torch.zeros((1, token_len, max_atoms_per_token, 4, 64),
                                    device=ref_atom_name_chars.device,
                                    dtype=ref_atom_name_chars.dtype)
    for token_idx in range(token_len):  
        # get the atom indices for the current token
        mask = (atom_token_indices[0] == token_idx) 
        current_atom_chars = ref_atom_name_chars[0, mask] 
        num_atoms = current_atom_chars.shape[0]
        if num_atoms > 0:
            pad_ref_atom_name_chars[0, token_idx, :num_atoms] = current_atom_chars

    pad_ref_atom_name_chars = F.pad(pad_ref_atom_name_chars, 
                                    (0, 0, 0, 0, 0, 0, 0, padded_token_length - token_len), 
                                    'constant', 0)


    ref_space_uid = boltz_input_features['ref_space_uid'][atom_pad_mask==1].unsqueeze(0)  
    pad_ref_space_uid = torch.zeros((1, token_len, max_atoms_per_token),
                                device=ref_space_uid.device,
                                dtype=ref_space_uid.dtype)  # 默认全0

    for token_idx in range(token_len):
        # Get all atom indices for current token
        mask = (atom_token_indices[0] == token_idx)
        if mask.any():  # if there are atoms for this token
            first_atom_uid = ref_space_uid[0, mask.nonzero()[0, 0]]  # get the first atom uid
            pad_ref_space_uid[0, token_idx, :] = first_atom_uid  # fill the features with the first atom uid
    pad_ref_space_uid = F.pad(pad_ref_space_uid, 
                                (0, 0, 0, padded_token_length - token_len), 
                                'constant', 0)
    
    #### ref_mask
    pad_ref_mask = torch.zeros((1, token_len, max_atoms_per_token), 
                        device=ref_charge.device, 
                        dtype=torch.bool)
    for token_idx in range(token_len):
        # Get all atom indices for current token
        mask = (atom_token_indices[0] == token_idx)  
        current_charges = ref_charge[0, mask] 
        num_atoms = current_charges.shape[0]
        
        # If current token has atoms, fill the features
        if num_atoms > 0:
            pad_ref_mask[0, token_idx, :num_atoms] = True
    pad_ref_mask = F.pad(pad_ref_mask, (0, 0, 0, padded_token_length - token_len), 'constant', 0)
    
    input_features['ref_pos'] = pad_ref_pos
    input_features['ref_element'] = pad_ref_element.float()
    input_features['ref_charge'] = pad_ref_charge.float()
    input_features['ref_atom_name_chars'] = pad_ref_atom_name_chars.float()
    input_features['ref_space_uid'] = pad_ref_space_uid.int()
    input_features['ref_mask'] = pad_ref_mask
    
    
    pad_pred_dense_atom_mask = torch.zeros((1, token_len, max_atoms_per_token), 
                                dtype=torch.bool,
                                device=atom_counts_per_token.device)
    for token_idx in range(token_len):
        num_atoms = atom_counts_per_token[0, token_idx]
        if num_atoms > 0:
            pad_pred_dense_atom_mask[0, token_idx, :num_atoms] = True
    pad_pred_dense_atom_mask = F.pad(
        pad_pred_dense_atom_mask,
        (0, 0, 0, padded_token_length - token_len), 
        mode='constant',
        value=False
    )
    input_features['pred_dense_atom_mask'] = pad_pred_dense_atom_mask
    
    ###### residue_center_index #########  
    atom_idx = boltz_input_features['atom_idx']
    center_idx = boltz_input_features['center_idx']
    residue_center_index = center_idx - atom_idx
    residue_center_index = F.pad(residue_center_index, 
                (0, padded_token_length - token_len), 
                'constant', 0)
    input_features['residue_center_index'] = residue_center_index
    ###### residue_center_index #########  
    
    
    ###### atom_pseudo_beta_index #########  
    token_to_rep_atom = boltz_input_features['token_to_rep_atom']
    valid_atom_idx = torch.where(atom_pad_mask==1)[1]   
    token_to_rep_atom = token_to_rep_atom[:, :, valid_atom_idx]

    atom_nums = boltz_input_features['atom_nums']
    atom_accumulated = 0
    atom_pseudo_beta_index = torch.zeros((1, token_len), device=atom_pad_mask.device, dtype=torch.int64)
    for i, token_idx in enumerate(range(token_len)):
        cur_token_atom_nums = int(atom_nums[0][token_idx])
        start_atom_idx = 24*i
        cur_atom_pseudo_beta_index = torch.where(token_to_rep_atom[0][token_idx] == 1)[0]-atom_accumulated
        atom_pseudo_beta_index[0,token_idx] = cur_atom_pseudo_beta_index + start_atom_idx
        atom_accumulated += cur_token_atom_nums
    ## pad
    atom_pseudo_beta_index = F.pad(atom_pseudo_beta_index,
                        (0, padded_token_length - token_len), 
                        'constant', 0)
    input_features['atom_pseudo_beta_index'] = atom_pseudo_beta_index
    ###### atom_pseudo_beta_index #########  

    pseudo_beta_mask = torch.ones((1, token_len), device=atom_pad_mask.device, dtype=torch.bool)
    ## pad
    pseudo_beta_mask = F.pad(pseudo_beta_mask,
                        (0, padded_token_length - token_len), 
                        'constant', False)
    input_features['pseudo_beta_mask'] = pseudo_beta_mask
    
    frame_mask = ~boltz_input_features['frame_resolved_mask']
    ## pad
    frame_mask = F.pad(frame_mask,
                        (0, padded_token_length - token_len), 
                        'constant', False)
    input_features['frame_mask'] = frame_mask


    # ### empty template feature
    # # 'template_aatype', 'template_distogram', 'template_pseudo_beta_mask', 'template_unit_vector', 'template_backbone_frame_mask'
    # # template_aatype: [bs, n_templ, num_token, 31]
    # template_aatype = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1]))
    # template_aatype = F.one_hot(template_aatype.long(),num_classes=31).float()
    # # template_distogram: [bs, n_templ, num_token, num_token, 39]
    # template_distogram = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1], input_features['aatype'].shape[1], 39))
    # # template_pseudo_beta_mask: [bs, n_templ, num_token]
    # template_pseudo_beta_mask = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1]))
    # # template_unit_vector: [bs, n_templ, num_token, num_token, 3]
    # template_unit_vector = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1], input_features['aatype'].shape[1], 3))
    # # template_backbone_frame_mask [bs, n_templ, num_token]
    # template_backbone_frame_mask = torch.zeros((input_features['aatype'].shape[0], 4, input_features['aatype'].shape[1]))
            
    # input_features['template_aatype'] = template_aatype.float()
    # input_features['template_distogram'] = template_distogram.float()
    # input_features['template_pseudo_beta_mask'] = template_pseudo_beta_mask.float()
    # input_features['template_unit_vector'] = template_unit_vector.float()
    # input_features['template_backbone_frame_mask'] = template_backbone_frame_mask.float()
    
    input_features['record'] = boltz_input_features['record']
    input_features['structure'] = boltz_input_features['structure']
    
    input_features['msa'] = input_features['msa'].argmax(dim=-1)
    
    #### summary infomation
    input_features['N_chains'] = input_features['asym_id'].max()
    input_features['N_tokens'] = torch.tensor(total_tokens)
    input_features['N_atoms'] = input_features['atom_nums'].sum()
    input_features['N_alignments'] = input_features['num_alignments'].squeeze(0)
    
    return input_features
        
def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]
        
        if key == 'structure':
            collated[key] = values[0]
            continue
        
        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
        ]:
            # Check if all have the same sha哦跑；pe
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values
    collated = transform(collated)
    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        constraints_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        record = self.manifest.records[idx]
        # Get the structure
        try:
            input_data = load_input(
                record,
                self.target_dir,
                self.msa_dir,
                self.constraints_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)
        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None:
            binders, pocket = None, None
        else:
            binders, pocket = options.binders, options.pocket

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
                compute_constraint_features=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)
        features["record"] = record
        features['structure'] = input_data.structure
        
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


def get_inference_dataloader(
    args,
    manifest: Manifest,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
) -> DataLoader:
    """Get the inference dataloader.

    Parameters
    ----------
    args : Namespace
        The command line arguments.
    manifest : Manifest
        The manifest to load data from.
    target_dir : Path
        The path to the target directory.
    msa_dir : Path
        The path to the msa directory.
    constraints_dir : Optional[Path], optional
        The path to the constraints directory, by default None

    Returns
    -------
    DataLoader
        The inference dataloader.

    """
    dataset = PredictionDataset(
        manifest=manifest,
        target_dir=target_dir,
        msa_dir=msa_dir,
        constraints_dir=constraints_dir,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate,
        )
    
    return dataloader