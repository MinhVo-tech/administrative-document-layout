"""Write dataset/splits/<split>.txt: one line per page image, '<image file>\t<document id>'.

    python -m dataset.tools.make_splits
"""
import argparse
import os

from common.config import repo_path
from common.constants import SPLITS
from dataset.tools.naming import doc_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="dataset/labels")
    ap.add_argument("--out", default="dataset/splits")
    args = ap.parse_args()
    out = repo_path(args.out)
    os.makedirs(out, exist_ok=True)
    for split in SPLITS:
        names = sorted(f[:-4] + ".jpg" for f in os.listdir(repo_path(args.labels) / split) if f.endswith(".txt"))
        with open(out / f"{split}.txt", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(f"{n}\t{doc_id(n)}\n" for n in names)
        print(f"{split}: {len(names)} images, {len({doc_id(n) for n in names})} documents -> {out / f'{split}.txt'}")


if __name__ == "__main__":
    main()
