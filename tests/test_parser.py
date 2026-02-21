import pytest
from pathlib import Path
from citation_tracker.parser import extract_text

def test_extract_text_mock(mocker, tmp_path):
    # Mocking pymupdf4llm
    mock_pymupdf = mocker.patch("pymupdf4llm.to_markdown")
    mock_pymupdf.return_value = "Extracted Text"
    
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy content")
    
    text = extract_text(pdf_path)
    assert text == "Extracted Text"
    mock_pymupdf.assert_called_once_with(str(pdf_path))

def test_extract_text_failure(mocker, tmp_path):
    mock_pymupdf = mocker.patch("pymupdf4llm.to_markdown")
    mock_pymupdf.side_effect = Exception("Parsing error")
    
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy content")
    
    text = extract_text(pdf_path)
    assert text is None
