from lima_llm.chunking.utils import complement_chunk_ids, compose_text_from_chunk_ids
from lima_llm.types import TextChunk


def test_subset_and_complement_are_disjoint_and_cover_universe() -> None:
    chunks = [
        TextChunk(chunk_id=0, start_char=0, end_char=2, text="ab"),
        TextChunk(chunk_id=1, start_char=2, end_char=4, text="cd"),
        TextChunk(chunk_id=2, start_char=4, end_char=6, text="ef"),
    ]
    selected = [0, 2]
    comp = complement_chunk_ids(chunks, selected)

    assert set(selected).isdisjoint(set(comp))
    assert set(selected).union(set(comp)) == {0, 1, 2}

    full = compose_text_from_chunk_ids(chunks, [0, 1, 2])
    selected_text = compose_text_from_chunk_ids(chunks, selected)
    comp_text = compose_text_from_chunk_ids(chunks, comp)

    assert full == "abcdef"
    assert selected_text == "abef"
    assert comp_text == "cd"
