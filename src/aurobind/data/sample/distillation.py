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

from typing import Iterator, List

from numpy.random import RandomState

from aurobind.data.types import Record
from aurobind.data.sample.sampler import Sample, Sampler


class DistillationSampler(Sampler):
    """A sampler for monomer distillation data."""

    def __init__(self, small_size: int = 200, small_prob: float = 0.01) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        small_size : int, optional
            The maximum size to be considered small.
        small_prob : float, optional
            The probability of sampling a small item.

        """
        self._size = small_size
        self._prob = small_prob

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:
        """Sample a structure from the dataset infinitely.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample.

        """
        # Remove records with invalid chains
        records = [r for r in records if r.chains[0].valid]

        # Split in small and large proteins. We assume that there is only
        # one chain per record, as is the case for monomer distillation
        small = [r for r in records if r.chains[0].num_residues <= self._size]
        large = [r for r in records if r.chains[0].num_residues > self._size]

        # Sample infinitely
        while True:
            # Sample small or large
            samples = small if random.rand() < self._prob else large

            # Sample item from the list
            index = random.randint(0, len(samples))
            yield Sample(record=samples[index])
