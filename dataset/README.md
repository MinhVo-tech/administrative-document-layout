# VBHC dataset: field annotations for Vietnamese administrative documents

Bounding-box annotations of seven information fields on the first and last pages of Vietnamese administrative
documents, the corresponding page images, the test PDFs and the text ground truth used to evaluate OCR.

License: [CC BY-NC 4.0](LICENSE), **non-commercial research only** (see [Intended use](#intended-use)).

## Contents

| Path | Content | Where |
|---|---|---|
| `labels/<split>/*.txt` | YOLO labels: `class x_center y_center width height` (normalised) | this repository |
| `annotations/voc/<split>/*.xml` | the same boxes in Pascal VOC format (pixels) | this repository |
| `annotations/ocr_gt.json` | text of the seven fields for the 154 test documents | this repository |
| `splits/<split>.txt` | `<image file>\t<document id>` | this repository |
| `images/<split>/*.jpg` | 2,720 page images | Zenodo (`images.zip`) |
| `pdf-test/<Type>/*.pdf` | source PDFs of the 154 test documents | Zenodo (`pdf-test.zip`) |
| `tools/` | consistency checks, PDF type analysis, release packaging | this repository |

## Fields

| id | Field | Description |
|---|---|---|
| 0 | `code` | document number and symbol, e.g. "Số: 16/CT-TTg" |
| 1 | `documentTitle` | title / summary of the content (may span several lines → several boxes) |
| 2 | `documentType` | type name printed on the document, e.g. "CHỈ THỊ" (absent on Dispatches) |
| 3 | `issuanceDate` | place and date, e.g. "Hà Nội, ngày 31 tháng 3 năm 2020" |
| 4 | `issuingAgency` | issuing authority (often two lines → two boxes) |
| 5 | `signatory` | name of the signer |
| 6 | `signatoryPosition` | position of the signer |

## File names

```
Chi-thi_012-0-2_front_0_png_jpg.rf.<hash>.jpg
└─type─┘└doc┘└──────page──────┘   └─Roboflow export suffix─┘
```

* `<Type>_<NNN>` is the **document id** = the stem of the source PDF (`Chi-thi_012@0@2.pdf`, `Cong-Van_241.pdf`).
* Type folders: `Quyet-dinh` (Decision), `Nghi-Quyet` (Resolution), `Chi-thi` (Directive), `Cong-Van`
  (Dispatch), `Cong-Dien` (Telegram), `ThongBao` (Notice), `Kehoach` (Plan). The prefixes `QDThongBao` and
  `Quyet-dinh-Cong-Van` are Decisions (`QDThongBao_192` was collected with a Notice; `Quyet-dinh-Cong-Van_241`
  was attached to Dispatch `Cong-Van_241` and split into its own file).
* `front` = first page, `last` = last page, `front_last` = single-page document; the number is the page index.

## OCR ground truth

`annotations/ocr_gt.json` maps each of the 154 test documents (document id) to the text of its fields:

```json
"Chi-thi_012": [{"issuingAgency": "THỦ TƯỚNG CHÍNH PHỦ"}, {"code": "Số: 16/CT-TTg"}, ...]
```

Fields that do not appear on a document (e.g. `documentType` on Dispatches) are omitted.

## Checks

```bash
python -m dataset.tools.check_pdf_yolo --pdf-root dataset/pdf-test --split test
```

## Intended use

The dataset is released for **non-commercial research** on document layout analysis and information
extraction (CC BY-NC 4.0). The documents are official administrative documents; under Vietnamese law
(Intellectual Property Law, Art. 15) legal and administrative documents are not subject to copyright
protection, but they contain personal data of officials (names, signatures, seals). Do not use the data to
identify, profile or contact individuals, or to forge, alter or impersonate documents, signatures or seals.
If you find content that should be removed, contact TODO: e-mail.

## Citation

See the [main README](../README.md#citation).
