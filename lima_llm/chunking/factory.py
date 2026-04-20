from __future__ import annotations

from typing import Callable, List, Optional

from ..types import TextChunk
from .fixed_token import fixed_token_chunk
from .sentence import sentence_chunk
from .utils import validate_chunk_coverage


Chunker = Callable[[str], List[TextChunk]]


def build_chunker(
    method: str,
    tokenizer=None,
    fixed_token_size: int = 64,
) -> Chunker:
    method = method.lower().strip()

    if method == "fixed_token":
        return lambda text: fixed_token_chunk(text=text, token_size=fixed_token_size, tokenizer=tokenizer)

    if method == "sentence":

        def _chunk(text: str) -> List[TextChunk]:
            chunks = sentence_chunk(text)
            ok, _ = validate_chunk_coverage(text, chunks)
            if not ok or len(chunks) <= 1:
                chunks = fixed_token_chunk(text=text, token_size=fixed_token_size, tokenizer=tokenizer)
            return chunks

        return _chunk

    raise ValueError(f"Unsupported chunker: {method}")
