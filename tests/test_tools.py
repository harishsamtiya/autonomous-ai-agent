"""
Unit tests for the Autonomous AI Agent.
"""

import pytest
import json
import math
from tools import calculator, text_summarizer, json_analyzer, get_current_datetime


class TestCalculatorTool:
    """Tests for the calculator tool."""

    def test_basic_addition(self):
        result = calculator.invoke("2 + 3")
        assert "5" in result

    def test_exponentiation(self):
        result = calculator.invoke("2**10")
        assert "1024" in result

    def test_sqrt(self):
        result = calculator.invoke("math.sqrt(144)")
        assert "12" in result

    def test_invalid_expression(self):
        result = calculator.invoke("invalid_expr")
        assert "Error" in result

    def test_division(self):
        result = calculator.invoke("100 / 4")
        assert "25" in result


class TestTextSummarizer:
    """Tests for the text summarizer tool."""

    def test_short_text(self):
        text = "This is a short text."
        result = text_summarizer.invoke(text)
        assert "Summary" in result

    def test_long_text(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        result = text_summarizer.invoke(text)
        assert "5 sentences" in result


class TestJsonAnalyzer:
    """Tests for the JSON analyzer tool."""

    def test_valid_json(self):
        data = json.dumps({"name": "test", "value": 42})
        result = json_analyzer.invoke(data)
        assert "dict" in result

    def test_invalid_json(self):
        result = json_analyzer.invoke("not json")
        assert "Invalid" in result

    def test_json_list(self):
        data = json.dumps([1, 2, 3])
        result = json_analyzer.invoke(data)
        assert "list" in result


class TestDatetimeTool:
    """Tests for the datetime tool."""

    def test_returns_date(self):
        result = get_current_datetime.invoke("")
        assert "Current date" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
