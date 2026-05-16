# processing/tests/test_llm_interface.py
"""Tests for the LLM interface — focused on _strip_code_fences, the markdown
fence stripper that both complete_json() implementations rely on."""
from processing.llm_interface import _strip_code_fences


def test_plain_json_unchanged():
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_newline_delimited_fence():
    assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_inline_fence_no_newline():
    """Regression: the old idiom did text.split(chr(10), 1)[1], which raised
    IndexError when the fence had no newline after ```json."""
    assert _strip_code_fences('```json{"a": 1}```') == '{"a": 1}'


def test_bare_fence_no_language_tag():
    assert _strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_fence_with_trailing_prose():
    """Closing fence plus trailing model chatter is dropped."""
    assert _strip_code_fences('```json\n{"a": 1}\n```\nHope that helps!') == '{"a": 1}'


def test_no_fence_with_whitespace():
    assert _strip_code_fences('  {"a": 1}  ') == '{"a": 1}'
