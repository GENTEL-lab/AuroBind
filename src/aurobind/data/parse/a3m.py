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

import gzip
from pathlib import Path
from typing import Optional, TextIO

import numpy as np

from aurobind.data import const
from aurobind.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def _parse_a3m(  # noqa: C901
    lines: TextIO,
    taxonomy: Optional[dict[str, str]],
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an MSA file.

    Parameters
    ----------
    lines : TextIO
        The lines of the MSA file.
    taxonomy : dict[str, str]
        The taxonomy database, if available.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line in lines:
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue

        # Get taxonomy, if annotated
        if line.startswith(">"):
            header = line.split()[0]
            if taxonomy and header.startswith(">UniRef100"):
                uniref_id = header.split("_")[1]
                taxonomy_id = taxonomy.get(uniref_id)
                if taxonomy_id is None:
                    taxonomy_id = -1
            else:
                taxonomy_id = -1
            continue

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Process sequence
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            token = const.prot_letter_to_token[c]
            # token = const.token_ids[token]
            token=const.mapping_boltz_token_ids_to_our_token_ids[const.token_ids[token]]
            residue.append(token)
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa


def parse_a3m(
    path: Path,
    taxonomy: Optional[dict[str, str]],
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    taxonomy : Redis
        The taxonomy database.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read the file
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)
    else:
        with path.open("r") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)
    return msa
