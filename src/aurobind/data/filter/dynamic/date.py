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

from datetime import datetime
from typing import Literal

from aurobind.data.types import Record
from aurobind.data.filter.dynamic.filter import DynamicFilter


class DateFilter(DynamicFilter):
    """A filter that filters complexes based on their date.

    The date can be the deposition, release, or revision date.
    If the date is not available, the previous date is used.

    If no date is available, the complex is rejected.

    """

    def __init__(
        self,
        date: str,
        ref: Literal["deposited", "revised", "released"],
    ) -> None:
        """Initialize the filter.

        Parameters
        ----------
        date : str, optional
            The maximum date of PDB entries to filter
        ref : Literal["deposited", "revised", "released"]
            The reference date to use.

        """
        self.filter_date = datetime.fromisoformat(date)
        self.ref = ref

        if ref not in ["deposited", "revised", "released"]:
            msg = (
                "Invalid reference date. Must be ",
                "deposited, revised, or released",
            )
            raise ValueError(msg)

    def filter(self, record: Record) -> bool:
        """Filter a record based on its date.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        structure = record.structure

        if self.ref == "deposited":
            date = structure.deposited
        elif self.ref == "released":
            date = structure.released
            if not date:
                date = structure.deposited
        elif self.ref == "revised":
            date = structure.revised
            if not date and structure.released:
                date = structure.released
            elif not date:
                date = structure.deposited

        if date is None or date == "":
            return False

        date = datetime.fromisoformat(date)
        return date <= self.filter_date
