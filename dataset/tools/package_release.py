"""Build the archives uploaded to Zenodo and their SHA-256 checksums.

    python -m dataset.tools.package_release --images path/to/vbhc-edited-1 --pdf-test path/to/pdf-test \
        --models path/to/YL02.pt path/to/F02.pth path/to/Re03.pth --out dist

Creates in --out:
  images.zip    images/{train,valid,test}/*.jpg       (unzip into dataset/)
  pdf-test.zip  pdf-test/<Type>/*.pdf                  (unzip into dataset/)
  YL02.pt, F02.pth, Re03.pth                           (copied; place them in models/)
  SHA256SUMS.txt
The image and PDF names are checked against the labels of this repository before zipping.
"""
import argparse
import hashlib
import os
import shutil
import zipfile

from common.config import repo_path
from common.constants import IMG_EXTS, SPLITS


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_split_images(root, split):
    """Images of one split in <root>/<split>/images (Roboflow YOLO export) or <root>/<split>."""
    for d in (os.path.join(root, split, "images"), os.path.join(root, split)):
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]
            if files:
                return d, files
    raise FileNotFoundError(f"no images for split '{split}' under {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Roboflow export with <split>/images/*.jpg")
    ap.add_argument("--pdf-test", required=True, help="folder with <Type>/*.pdf (154 test PDFs)")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    outputs = []

    zpath = os.path.join(args.out, "images.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress further
        for split in SPLITS:
            src, files = find_split_images(args.images, split)
            labels = {f[:-4] for f in os.listdir(repo_path("dataset/labels") / split) if f.endswith(".txt")}
            stems = {os.path.splitext(f)[0] for f in files}
            if stems != labels:
                raise ValueError(f"{split}: images and labels differ "
                                 f"({len(stems - labels)} images without label, {len(labels - stems)} labels without image)")
            for f in sorted(files):
                z.write(os.path.join(src, f), f"images/{split}/{f}")
            print(f"{split}: {len(files)} images")
    outputs.append(zpath)

    zpath = os.path.join(args.out, "pdf-test.zip")
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(args.pdf_test):
            for f in sorted(files):
                if f.lower().endswith(".pdf"):
                    z.write(os.path.join(d, f), f"pdf-test/{os.path.basename(d)}/{f}")
                    n += 1
    print(f"pdf-test: {n} PDFs")
    outputs.append(zpath)

    for m in args.models:
        dst = os.path.join(args.out, os.path.basename(m))
        shutil.copy2(m, dst)
        outputs.append(dst)

    with open(os.path.join(args.out, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        for p in outputs:
            digest = sha256(p)
            f.write(f"{digest}  {os.path.basename(p)}\n")
            print(f"{digest}  {os.path.basename(p)}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    print(f"Archives and SHA256SUMS.txt written to {args.out}")


if __name__ == "__main__":
    main()
