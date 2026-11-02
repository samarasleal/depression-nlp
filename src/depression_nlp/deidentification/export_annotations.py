"""Export selected, deidentified utterances to an annotation workbook."""

import argparse
import json
import re
from pathlib import Path
import pandas as pd


OUTPUT_COLUMNS = [
    "Patient_code",
    "Therapist_code",
    "Session",
    "Utterance_id",
    "Utterance",
    "Label",
    "Exclude_sensitive",
]


def load_reference(spreadsheet_path: str | Path) -> pd.DataFrame:
    """Load patient and therapist anonymous codes."""
    df = pd.read_excel(spreadsheet_path)

    required = {"Patient_ID", "Therapist_ID", "Patient_code", "Therapist_code"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing spreadsheet columns: {', '.join(sorted(missing))}"
        )

    return df


def get_anonymous_codes(
    reference_df: pd.DataFrame,
    patient_id: str,
    therapist_id: str,
) -> tuple[str, str]:
    """Find anonymous codes using the patient-therapist reference row."""
    rows = reference_df[
        (reference_df["Patient_ID"].astype(str) == str(patient_id))
        & (reference_df["Therapist_ID"].astype(str) == str(therapist_id))
    ]

    if rows.empty:
        raise ValueError(
            f"Codes not found for patient {patient_id} and therapist {therapist_id}."
        )

    row = rows.iloc[0]
    patient_code = row["Patient_code"]
    therapist_code = row["Therapist_code"]

    if pd.isna(patient_code) or pd.isna(therapist_code):
        raise ValueError(
            f"Anonymous codes are empty for patient {patient_id}."
        )

    return str(patient_code), str(therapist_code)


def create_sheet_name(patient_code: str, therapist_code: str) -> str:
    """Create a valid Excel sheet name."""
    name = f"{patient_code}_{therapist_code}"
    return re.sub(r"[\[\]:*?/\\]", "_", name)[:31]


def collect_selected_utterances(
    input_root: str | Path,
    reference_df: pd.DataFrame,
) -> list[dict]:
    """Collect selected utterances from all deidentified JSON files."""
    rows = []

    for json_path in sorted(Path(input_root).rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        patient_id = str(payload["patient_id"])
        therapist_id = str(payload["therapist_id"])
        session = int(payload["session"])

        patient_code, therapist_code = get_anonymous_codes(
            reference_df,
            patient_id,
            therapist_id,
        )

        for utterance in payload.get("utterances", []):
            if not utterance.get("selected_for_annotation", False):
                continue

            rows.append(
                {
                    "Patient_code": patient_code,
                    "Therapist_code": therapist_code,
                    "Session": session,
                    "Utterance_id": utterance["utterance_id"],
                    "Utterance": utterance["text"],
                    "Label": "",
                    "Exclude_sensitive": "",
                }
            )

    return rows


def save_annotation_workbook(
    rows: list[dict],
    output_path: str | Path,
) -> str:
    """Save one Excel sheet for each patient-therapist dyad."""
    if not rows:
        raise ValueError("No selected utterances were found.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    dyads = df.groupby(["Patient_code", "Therapist_code"], sort=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for (patient_code, therapist_code), dyad_df in dyads:
            sheet_name = create_sheet_name(
                str(patient_code),
                str(therapist_code),
            )

            dyad_df = dyad_df.sort_values(
                ["Session", "Utterance_id"]
            )

            dyad_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            worksheet.column_dimensions["A"].width = 18
            worksheet.column_dimensions["B"].width = 18
            worksheet.column_dimensions["C"].width = 10
            worksheet.column_dimensions["D"].width = 15
            worksheet.column_dimensions["E"].width = 90
            worksheet.column_dimensions["F"].width = 30
            worksheet.column_dimensions["G"].width = 20

            for cell in worksheet["E"]:
                cell.alignment = cell.alignment.copy(
                    wrap_text=True,
                    vertical="top",
                )

    return str(output_path.resolve())


def export_annotation_candidates(
    input_root: str | Path,
    reference_spreadsheet: str | Path,
    output_path: str | Path,
) -> str:
    """Run the complete annotation workbook export."""
    reference_df = load_reference(reference_spreadsheet)
    rows = collect_selected_utterances(input_root, reference_df)
    return save_annotation_workbook(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export selected utterances for clinical annotation."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing deidentified JSON files.",
    )
    parser.add_argument(
        "--spreadsheet",
        required=True,
        help="Reference spreadsheet containing anonymous codes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .xlsx file.",
    )
    args = parser.parse_args()

    try:
        output = export_annotation_candidates(
            input_root=args.input_root,
            reference_spreadsheet=args.spreadsheet,
            output_path=args.output,
        )
        print(f"Annotation workbook saved to {output}")
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Annotation export failed: {error}") from None


if __name__ == "__main__":
    main()