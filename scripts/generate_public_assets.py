"""Generate schemas and a non-patient synthetic smoke-test input."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CLINICAL = [
    ("age", "Age", "年龄", "continuous", "years", "Nonnegative numeric value"),
    ("sex", "Sex", "性别", "categorical", "", "1=male; 2=female"),
    ("differentiation", "Histologic differentiation", "组织学分化程度", "categorical", "", "1=well; 2=moderate; 3=poor"),
    ("tumor_size_cm", "Maximum tumor diameter", "肿瘤最大径", "continuous", "cm", "Nonnegative numeric value"),
    ("distance_from_anal_verge_cm", "Distance from anal verge", "肿瘤距肛缘距离", "continuous", "cm", "Nonnegative numeric value"),
    ("margin_status", "Resection margin status", "切缘状态", "categorical", "", "0=negative; 1=positive"),
    ("pni", "Perineural invasion", "神经侵犯", "categorical", "", "0=absent; 1=present"),
    ("lvi", "Lymphovascular invasion", "脉管侵犯", "categorical", "", "0=absent; 1=present"),
    ("pt_stage", "Pathologic T category", "病理T分期", "categorical", "AJCC/UICC 8th", "0, 1, 2, 3, 4, 4a, or 4b"),
    ("pn_stage", "Pathologic N category", "病理N分期", "categorical", "AJCC/UICC 8th", "0, 1a, 1b, 2a, or 2b"),
    ("examined_node_count", "Examined lymph-node count", "检及淋巴结总数", "continuous", "nodes", "Nonnegative integer"),
    ("cea", "Carcinoembryonic antigen", "癌胚抗原", "continuous", "ng/mL", "Nonnegative numeric value"),
    ("ca19_9", "Carbohydrate antigen 19-9", "糖类抗原19-9", "continuous", "U/mL", "Nonnegative numeric value"),
]


def main() -> None:
    bundle = joblib.load(ROOT / "models" / "m12_preprocessing.joblib")
    schema_dir = ROOT / "schemas"
    example_dir = ROOT / "examples"
    schema_dir.mkdir(exist_ok=True)
    example_dir.mkdir(exist_ok=True)

    with (schema_dir / "clinical_variables.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "english_label", "chinese_label", "type", "unit_or_standard", "allowed_values"])
        writer.writerows(CLINICAL)

    (schema_dir / "radiomics_spatial_features.txt").write_text(
        "\n".join(bundle["radiomics_features"]) + "\n", encoding="utf-8"
    )
    (schema_dir / "cnn_features.txt").write_text(
        "\n".join(bundle["cnn_features"]) + "\n", encoding="utf-8"
    )
    machine_schema = {
        "clinical": [row[0] for row in CLINICAL],
        "radiomics_spatial": bundle["radiomics_features"],
        "cnn": bundle["cnn_features"],
        "total_input_columns": len(CLINICAL) + len(bundle["radiomics_features"]) + len(bundle["cnn_features"]),
    }
    (schema_dir / "input_schema.json").write_text(
        json.dumps(machine_schema, indent=2), encoding="utf-8"
    )

    synthetic = {
        "age": 60,
        "sex": 1,
        "differentiation": 2,
        "tumor_size_cm": 4.0,
        "distance_from_anal_verge_cm": 7.0,
        "margin_status": 0,
        "pni": 0,
        "lvi": 0,
        "pt_stage": "3",
        "pn_stage": "0",
        "examined_node_count": 15,
        "cea": 3.0,
        "ca19_9": 12.0,
    }
    synthetic.update(dict(zip(bundle["radiomics_features"], bundle["radiomics_imputer"].statistics_.tolist())))
    synthetic.update(dict(zip(bundle["cnn_features"], bundle["cnn_imputer"].statistics_.tolist())))
    pd.DataFrame([synthetic]).to_csv(example_dir / "synthetic_input.csv", index=False)


if __name__ == "__main__":
    main()
