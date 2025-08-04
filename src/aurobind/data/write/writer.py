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

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from aurobind.data.types import (
    Interface,
    Record,
    Structure,
)
from aurobind.data.write.mmcif import to_mmcif
from aurobind.data.write.pdb import to_pdb


def write_cif(structure, record, coord, plddts, output_path, output_format='mmcif'):
    # Compute chain map with masked removed, to be used later
    chain_map = {}
    for i, mask in enumerate(structure.mask):
        if mask:
            chain_map[len(chain_map)] = i
    # Remove masked chains completely
    structure = structure.remove_invalid_chains()
    # for model_idx in range(coord.shape[0]):
    # # Get model coord
    model_coord = coord.squeeze(0)
    model_plddts = plddts.squeeze(0)

    # New atom table
    atoms = structure.atoms
    # atoms["coords"] = coord_unpad
    atoms["coords"] = model_coord
    atoms["is_present"] = True

    # Mew residue table
    residues = structure.residues
    residues["is_present"] = True

    # Update the structure
    interfaces = np.array([], dtype=Interface)
    new_structure: Structure = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
    )

    # Update chain info
    chain_info = []
    for chain in new_structure.chains:
        old_chain_idx = chain_map[chain["asym_id"]]
        old_chain_info = record.chains[old_chain_idx]
        new_chain_info = replace(
            old_chain_info,
            chain_id=int(chain["asym_id"]),
            valid=True,
        )
        chain_info.append(new_chain_info)

    # Save the structure
    if output_format == "pdb":
        with output_path.open("w") as f:
            f.write(to_pdb(new_structure, plddts=model_plddts))
    elif output_format == "mmcif":
        with output_path.open("w") as f:
            f.write(to_mmcif(new_structure, plddts=model_plddts))