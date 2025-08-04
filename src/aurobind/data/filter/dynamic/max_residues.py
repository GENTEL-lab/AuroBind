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

from aurobind.data.types import Record
from aurobind.data.filter.dynamic.filter import DynamicFilter


class MaxResiduesFilter(DynamicFilter):
    """A filter that filters structures based on their size."""

    def __init__(self, min_residues: int = 1, max_residues: int = 500) -> None:
        """Initialize the filter.

        Parameters
        ----------
        min_chains : int
            The minimum number of chains allowed.
        max_chains : int
            The maximum number of chains allowed.

        """
        self.min_residues = min_residues
        self.max_residues = max_residues

    def filter(self, record: Record) -> bool:
        """Filter structures based on their resolution.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        num_residues = sum(chain.num_residues for chain in record.chains)
        return num_residues <= self.max_residues and num_residues >= self.min_residues
