"""Classify source PDFs as text-based, scanned image-based or mixed (dataset description).

    python -m dataset.tools.pdf_types --pdf-root path/to/vbhc-pdf-clean      # requires PyMuPDF

Page classes:
  text      >= MIN_CHARS visible extractable characters (born-digital page)
  scan      embedded images cover >= IMG_COVER of the page and no OCR text layer
  scan_ocr  embedded images cover >= IMG_COVER of the page with an invisible OCR text layer
            (text render mode 3 or opacity 0) of >= MIN_CHARS characters
  blank     no text and no significant image (ignored)
A PDF is text-based if all of its (non-blank) pages have a text layer, scanned if none has, mixed otherwise.
Both readings of OCR'd scans are reported: counted as text layer (used in the paper) or as scanned.
"""
import argparse
import os

MIN_CHARS = 50
IMG_COVER = 0.5
GRID = 100  # image coverage is measured on a GRID x GRID point grid (union of the image rectangles)


def image_cover(page):
    import pymupdf

    rects = [pymupdf.Rect(i["bbox"]) & page.rect for i in page.get_image_info()]
    rects = [r for r in rects if not r.is_empty]
    if not rects:
        return 0.0
    W, H = page.rect.width, page.rect.height
    covered = 0
    for gy in range(GRID):
        y = page.rect.y0 + (gy + 0.5) * H / GRID
        for gx in range(GRID):
            x = page.rect.x0 + (gx + 0.5) * W / GRID
            covered += any(r.contains(pymupdf.Point(x, y)) for r in rects)
    return covered / (GRID * GRID)


def char_counts(page):
    """(visible characters, invisible characters) of the text layer."""
    visible = hidden = 0
    for span in page.get_texttrace():
        n = sum(1 for c in span["chars"] if chr(c[0]).strip())
        if span.get("type") == 3 or span.get("opacity", 1) == 0:
            hidden += n
        else:
            visible += n
    return visible, hidden


def page_type(page):
    cover = image_cover(page)
    visible, hidden = char_counts(page)
    if visible >= MIN_CHARS:
        return "text"
    if cover >= IMG_COVER:
        return "scan_ocr" if hidden >= MIN_CHARS else "scan"
    if cover > 0 and visible + hidden < MIN_CHARS:
        return "scan"
    return "blank"


def classify(path):
    import pymupdf

    rec = {"file": path, "pages": 0, "text": 0, "scan": 0, "scan_ocr": 0, "blank": 0, "page_errors": 0, "error": ""}
    try:
        with pymupdf.open(path) as doc:
            rec["pages"] = len(doc)
            for p in doc:
                try:
                    rec[page_type(p)] += 1
                except Exception:
                    rec["page_errors"] += 1
    except Exception as e:
        rec["error"] = str(e)[:80]
    return rec


def label(r, ocr_as_text):
    if r["error"] or r["pages"] == r["page_errors"]:
        return "unreadable"
    t = r["text"] + (r["scan_ocr"] if ocr_as_text else 0)
    s = r["scan"] + (0 if ocr_as_text else r["scan_ocr"])
    if t and s:
        return "mixed"
    return "scanned" if s else ("text" if t else "blank")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-root", required=True)
    ap.add_argument("--out", default="outputs/pdf_types.csv")
    args = ap.parse_args()

    import pandas as pd
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
    pdfs = sorted(os.path.join(d, f) for d, _, fs in os.walk(args.pdf_root) for f in fs if f.lower().endswith(".pdf"))
    df = pd.DataFrame([classify(p) for p in pdfs])
    print(f"{len(df)} PDFs | unreadable: {(df.error != '').sum()} | page errors: {df.page_errors.sum()} | "
          f"blank pages ignored: {df.blank.sum()}")
    print(f"pages: text={df.text.sum()}, scan={df.scan.sum()}, scan with OCR layer={df.scan_ocr.sum()}\n")
    for ocr_as_text, title in [(True, "OCR'd scans count as text layer (paper)"), (False, "OCR'd scans count as scanned")]:
        col = "type_ocr_as_text" if ocr_as_text else "type_ocr_as_scan"
        df[col] = df.apply(label, axis=1, ocr_as_text=ocr_as_text)
        c = df[col].value_counts()
        n = sum(c.get(k, 0) for k in ("text", "scanned", "mixed"))
        print(f"== {title} (n = {n}) ==")
        for k, name in [("text", "text-based"), ("scanned", "scanned image-based"), ("mixed", "mixed")]:
            print(f"  {name:<22}{c.get(k, 0):>6}  ({100 * c.get(k, 0) / n:.1f}%)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
