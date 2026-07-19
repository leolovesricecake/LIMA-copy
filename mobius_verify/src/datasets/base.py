from __future__ import annotations

from typing import Protocol, Sequence

from ..schema import TextRecord


class DatasetAdapter(Protocol):
    def load(self) -> Sequence[TextRecord]:
        ...

