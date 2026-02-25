"""Tests for dataset formatting functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.formatting import build_formatter


def test_medqa_formatter():
    formatter = build_formatter("bigbio/med_qa")
    examples = {
        "question": ["What is the most common cause of death in the US?"],
        "options": [["Heart disease", "Cancer", "Accidents", "COVID-19"]],
        "answer": [0],
    }
    result = formatter(examples)
    assert len(result) == 1
    assert "Heart disease" in result[0]
    assert "Question" in result[0]


def test_pubmedqa_formatter():
    template = "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    formatter = build_formatter("qiaojin/PubMedQA", template)
    examples = {
        "context": [["Study shows X.", "Another study confirms X."]],
        "question": ["Is X effective?"],
        "final_decision": ["yes"],
    }
    result = formatter(examples)
    assert len(result) == 1
    assert "yes" in result[0]
    assert "Study shows X." in result[0]


def test_generic_formatter():
    template = "Q: {question} A: {answer}"
    formatter = build_formatter("unknown/dataset", template)
    examples = {
        "question": ["What is 2+2?"],
        "answer": ["4"],
    }
    result = formatter(examples)
    assert result == ["Q: What is 2+2? A: 4"]
