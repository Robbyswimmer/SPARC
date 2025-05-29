# tests/test_length_schedule.py
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.curricula.length_schedule import LengthScheduleWrapper

from transformers import AutoTokenizer

_TOKENIZER_NAME = "NousResearch/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(
    _TOKENIZER_NAME,
    use_fast=True,
    add_eos_token=True
)

def dummy_iter(num_chunks=5):
    tpl = {
        "doc_chunks": [[i] for i in range(num_chunks)],
        "question_ids": [99],
        "answers": ["a"],
        "meta": {}
    }
    while True:
        yield tpl

@pytest.mark.parametrize("step,total,max0,maxf,expected",
    [(0, 10, 2, 6, 2),
     (5, 10, 2, 6, 4),
     (10,10, 2, 6, 6),
     (11,10, 2, 6, 6)])        # clamp to final max
def test_max_chunks_progress(step,total,max0,maxf,expected):
    it = LengthScheduleWrapper(dummy_iter(), max0, maxf, total)
    it._current_step = step
    assert it.get_current_max_chunks() == expected

def test_curriculum_unlock():
    from src.data.data_registry import CurriculumDataLoader
    cur = CurriculumDataLoader(
        tokenizer,
        dataset_order=["easy","hard"],
        qa_thresholds=[0.2],
        split="train",
        seed=0
    )
    # starts with only first
    assert cur.active_datasets == ["easy"]
    # score below threshold → no change
    cur.update_curriculum(0.1)
    assert cur.active_datasets == ["easy"]
    # exceed → unlock
    cur.update_curriculum(0.25)
    assert cur.active_datasets == ["easy","hard"]

