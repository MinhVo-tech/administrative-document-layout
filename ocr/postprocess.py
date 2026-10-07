"""Rule-based post-processing of the OCR output (Table 16 'After')."""
import re
from typing import Dict, List


def remove_so_prefix(s: str) -> str:
    """`code`: strip a leading 'Số' / 'Số:' (applied to prediction and ground truth), then trim."""
    return re.sub(r"^\s*Số:?\s*", "", s, flags=re.IGNORECASE).strip()


def normalize_date(text: str) -> str:
    """`issuanceDate`: rewrite as 'ngày D tháng M năm YYYY' when day, month and year are all found."""
    if text == "":
        return ""
    ngay = re.search(r"(?:ngày\s*)?(\d{1,2})", text)
    thang = re.search(r"tháng\s+(\d{1,2})", text)
    nam = re.search(r"năm\s+(\d{4})", text)
    if not (ngay and thang and nam):
        return text
    return f"ngày {ngay.group(1)} tháng {thang.group(1)} năm {nam.group(1)}"


def identity(s):
    return s


def flatten_labels(data: Dict[str, List[Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    """{doc: [{field: text}, ...]} -> {doc: {field: text}} (a later entry of the same field wins)."""
    out = {}
    for doc, fields in data.items():
        merged = {}
        for item in fields:
            merged.update(item)
        out[doc] = merged
    return out

