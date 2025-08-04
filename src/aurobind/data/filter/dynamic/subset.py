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

from aurobind.data.types import Record
from aurobind.data.filter.dynamic.filter import DynamicFilter


class SubsetFilter(DynamicFilter):
    """Filter a data record based on a subset of the data."""

    def __init__(self, subset: str, reverse: bool = False) -> None:
        """Initialize the filter.

        Parameters
        ----------
        subset : str
            The subset of data to consider, one per line.

        """
        with Path(subset).open("r") as f:
            subset = f.read().splitlines()

        self.subset = {s.lower() for s in subset}
        self.reverse = reverse

    def filter(self, record: Record) -> bool:
        """Filter a data record.

        Parameters
        ----------
        record : Record
            The object to consider filtering in / out.

        Returns
        -------
        bool
            True if the data passes the filter, False otherwise.

        """
        if self.reverse:
            return record.id.lower() not in self.subset
        else:  # noqa: RET505
            return record.id.lower() in self.subset
