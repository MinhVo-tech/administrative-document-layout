"""File-name conventions: page image / label <-> source document <-> document type.

Page images are named '<Prefix>_<NNN>...' (+ Roboflow's '.rf.<hash>' suffix), e.g.
    Chi-thi_001-0-3_front_0_png.rf.<hash>.jpg     (Directive no. 1, first page)
    Cong-Van_241_0_front_last_png.rf.<hash>.jpg   (Dispatch no. 241, a one-page document)
The source PDF of a document is '<Type folder>/<Prefix>_<NNN>[...].pdf'.
"""
import re

NAME_RE = re.compile(r"^([A-Za-z\-]+)_(\d+)")

# Prefixes that differ from their type folder:
#   QDThongBao_192          a Decision (1508/QD-VKSCB) collected from a Notice batch
#   Quyet-dinh-Cong-Van_241 a Decision (4964/QD-UBND) attached to Dispatch Cong-Van_241, split into its own file
PREFIX_TYPE = {"QDThongBao": "Quyet-dinh", "Quyet-dinh-Cong-Van": "Quyet-dinh"}


def parse(name):
    """(prefix, number as written, e.g. '012') of an image / label / PDF file name, or None."""
    m = NAME_RE.match(name.split(".rf.")[0])
    return (m.group(1), m.group(2)) if m else None


def doc_id(name):
    """Source document of a file, same as the PDF stem: 'Chi-thi_012', 'Quyet-dinh-Cong-Van_241', ..."""
    prefix, num = parse(name)
    return f"{prefix}_{num}"


def doc_type(name):
    """Type folder of a file: 'Chi-thi', ..., with PREFIX_TYPE applied."""
    prefix, _ = parse(name)
    return PREFIX_TYPE.get(prefix, prefix)
