"""Check that a folder of source PDFs matches the documents of the YOLO labels, name by name.

    python -m dataset.tools.check_pdf_yolo --pdf-root path/to/vbhc-pdf-clean       # all 1,513 documents
    python -m dataset.tools.check_pdf_yolo --pdf-root dataset/pdf-test --split test  # the 154 test PDFs

The PDF folder is <pdf-root>/<Type>/<Prefix>_<NNN>[...].pdf. Reports documents without PDF, PDFs without
labels, duplicated document ids, PDFs in the wrong type folder and the counts per type.
"""
import argparse
import os
from collections import Counter

from common.config import repo_path
from common.constants import SPLITS
from dataset.tools.naming import doc_id, doc_type, parse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-root", required=True)
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--split", default=None, choices=SPLITS, help="only the documents of one split")
    args = ap.parse_args()

    pdfs, bad_names, dup = {}, [], []
    for d, _, files in os.walk(repo_path(args.pdf_root)):
        for f in files:
            if not f.lower().endswith(".pdf"):
                continue
            if parse(f) is None:
                bad_names.append(os.path.join(d, f))
                continue
            key = doc_id(f)
            if key in pdfs:
                dup.append((key, pdfs[key][1], f))
            pdfs[key] = (os.path.basename(d), f)

    splits = [args.split] if args.split else SPLITS
    docs = {}
    for s in splits:
        for f in os.listdir(repo_path(args.labels) / s):
            if f.endswith(".txt"):
                docs.setdefault(doc_id(f), set()).add(s)

    no_pdf = sorted(set(docs) - set(pdfs))
    no_doc = sorted(set(pdfs) - set(docs))
    wrong_folder = sorted((k, folder, f) for k, (folder, f) in pdfs.items() if folder != doc_type(f))

    print(f"PDFs: {len(pdfs)}   documents in labels ({'/'.join(splits)}): {len(docs)}")
    print(f"documents without PDF ({len(no_pdf)}): {no_pdf[:20]}")
    print(f"PDFs without labels  ({len(no_doc)}): {no_doc[:20]}")
    for title, items in [("file names not matching <Prefix>_<NNN>", bad_names), ("duplicated document ids", dup),
                         ("PDFs in the wrong type folder", wrong_folder)]:
        if items:
            print(f"WARNING {title}: {items[:20]}")
    multi = sorted(k for k, s in docs.items() if len(s) > 1)
    print(f"documents with pages in more than one split: {len(multi)} {multi}")

    print("\nPDFs per type folder (PDF / labelled documents):")
    pdf_count = Counter(folder for folder, _ in pdfs.values())
    doc_count = Counter(doc_type(k) for k in docs)
    for t in sorted(set(pdf_count) | set(doc_count)):
        print(f"  {t:<12}{pdf_count[t]:>6}{doc_count[t]:>6}")
    ok = not (no_pdf or no_doc or bad_names or dup or wrong_folder)
    print("\nOK: PDFs and labels match" if ok else "\nMISMATCH: see above")


if __name__ == "__main__":
    main()
