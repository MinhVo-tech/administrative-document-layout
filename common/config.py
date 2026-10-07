"""Load YAML configs and resolve paths relative to the repository root."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def utf8_stdout():
    """Print Vietnamese text on consoles whose default encoding is not UTF-8 (e.g. Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load_yaml(path):
    import yaml  # installed with ultralytics

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(p):
    """Absolute path; relative paths are taken from the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_paths(path="configs/paths.yaml"):
    """configs/paths.yaml with every value turned into an absolute Path."""
    cfg = load_yaml(repo_path(path))

    def resolve(node):
        if isinstance(node, dict):
            return {k: resolve(v) for k, v in node.items()}
        return repo_path(node) if isinstance(node, str) else node

    return resolve(cfg)


def split_dirs(paths, split):
    """(images_dir, yolo_labels_dir, voc_xml_dir) of one split."""
    d = paths["dataset"]
    return d["images"] / split, d["labels"] / split, d["voc"] / split
