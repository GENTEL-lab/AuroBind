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

import numpy as np
import pandas as pd

from aurobind.data import const
from aurobind.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def parse_csv(
    path: Path,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read file
    data = pd.read_csv(path)

    # Check columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    # Create taxonomy mapping
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Get taxonomy, if annotated
        taxonomy_id = -1
        if (str(key) != "nan") and (key is not None) and (key != ""):
            taxonomy_id = key

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
