# Depression Outcome Prediction using NLP

Do therapist interventions adapt to the patient's evolving vocal state, and do longitudinal therapist–patient interaction patterns distinguish subsequent clinical outcomes during CBT?

## Objectives:
1 — Therapist adaptation: How does the therapist's intervention profile relate to the patient's vocal state within each session?

2 — Longitudinal dynamics: How do the patient's vocal state and the therapist's interventions relate to changes in the patient's vocal state in the following session?

3 — Clinical outcomes: Do these longitudinal interaction patterns differ between patients with favorable and unfavorable clinical outcomes?

###### A simple example using Patient P01: ######

Session 1 - A(S1): Extract the patient's acoustic characteristics, including prosody, voice intensity, and spectral features - we use only non-lexical speech information.

Session 1 - T(S1): Extract the therapist's CBT-related interventions from the speech transcript, such as guided discovery, cognitive restructuring, empathy, and homework.

Session 2 - A(S2): Measure how the patient's acoustic state has changed between Sessions 1 and 2.  ΔA = A(S2) − A(S1), considering T(S1).

This process continues across consecutive sessions, from S1 to S8, allowing us to investigate longitudinal therapist–patient interaction patterns and their relationship with clinical outcomes.

## Current pipeline

1. Load the patient/df_ref spreadsheet.
2. Run local speaker diarization with `pyannote/speaker-diarization-community-1`.
3. Use the manually annotated patient interval to select the real patient embedding.
4. Match that embedding to the speakers found in every session.
5. Extract patient and therapist waveforms from exclusive diarization turns.
6. Extract F0, RMS energy, and 3D LogMel features from five-second chunks.
7. Store F0/energy values and LogMel `.npy` paths in the output DataFrame.

## Requirements

- Python 3.10 or newer
- `ffmpeg` installed on the HPC
- Access accepted for `pyannote/speaker-diarization-community-1`
- A Hugging Face read token in the local `.env`

The Community-1 model runs locally after download; session audio is not sent to a hosted diarization API.

## Installation

```bash
python -m venv .venv # create
source .venv/bin/activate # activate
pip install -e .
cp .env.example .env
```

Edit `.env` with the private HPC paths and token, then run:

```bash
depression-features  # To run:  python -m depression_nlp.pipeline
```

The output DataFrame contains columns such as:

```text
F0_Patient_Session1_Segment0_mean
F0_Therapist_Session1_Segment0_std
Energy_Patient_Session1_Segment0
Energy_Therapist_Session1_Segment0
LogMel_Patient_Session1_Segment0
LogMel_Therapist_Session1_Segment0
```

Each LogMel path points to one `[time, mel_bins]` NumPy array. Loading all paths for one role/session creates the CNN input `[segments, time, mel_bins]`.

## HOW TO RUN

#### RUN the whole pipeline ####

#### 1. Extract acoustic features and therapist transcripts
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/audios/extracted_audios \
  --output /workspace/app/depression-results/frozen_features.pkl \
  --feature-root /workspace/app/depression-results/frozen_features \
  --transcription-root /workspace/app/depression-results/frozen_transcriptions \
  --config configs/pipeline.yaml

#### 2. Run CBT screening and select annotation candidates
depression-cbt-screening \
  --input-root /workspace/app/depression-results/frozen_transcriptions \
  --output-root /workspace/app/depression-results/frozen_screening \
  --config configs/cbt_screening.yaml

#### 3. Deidentify the screened utterances
depression-deidentify \
  --input-root /workspace/app/depression-results/frozen_screening \
  --output-root /workspace/app/depression-results/frozen_deidentified

#### 4. Export the annotation workbook
depression-annotation \
  --input-root /workspace/app/depression-results/frozen_deidentified \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --output /workspace/app/depression-results/annotation_candidates.xlsx

#### 4. Extract scores
python -m depression_nlp.model_evaluation.extract_predictions \
  --input-root /workspace/app/depression-results/frozen_deidentified \
  --reference-spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --output /workspace/app/depression-results/model_evaluation/model_predictions.csv \
  --config /workspace/app/depression-nlp/configs/pipeline.yaml

#### 5. Evaluate the LLM model
python -m depression_nlp.model_evaluation.evaluate_predictions \
  --annotations /workspace/app/depression-results/annotated_candidates.xlsx \
  --predictions /workspace/app/depression-results/model_evaluation/model_predictions.csv \
  --output-root /workspace/app/depression-results/model_evaluation/results

#### 6. Calculate Threshold
python -m depression_nlp.model_evaluation.calculate_threshold \
  --annotations /workspace/app/depression-results/annotated_candidates.xlsx \
  --predictions /workspace/app/depression-results/model_evaluation/model_predictions.csv \
  --output-root /workspace/app/depression-results/model_evaluation/threshold \
  --config /workspace/app/depression-nlp/configs/pipeline.yaml

#### 7. Get the therapist profile
python -m depression_nlp.feature_engineering.therapist_profile \
  --input-root /workspace/app/depression-results/frozen_deidentified \
  --reference-spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --output-root /workspace/app/depression-results/therapist_profiles \
  --config /workspace/app/depression-nlp/configs/pipeline.yaml

#### 8. Create the final dataset - patient's acoustic features + therapist profile + outcomes
python -m depression_nlp.feature_engineering.build_modeling_dataset \
  --features /workspace/app/depression-results/frozen_features.pkl \
  --profiles /workspace/app/depression-results/therapist_profiles/therapist_session_profiles.csv \
  --output-root /workspace/app/depression-results/modeling \
  --config configs/pipeline.yaml \
  --excel-pt

---

###### 1 -  How to test with only one patient - SVR01 ######
AUDIO_ROOT=/workspace/app/audios/test_one_patient
mkdir -p /workspace/app/test_one_patient
ln -s \
  /workspace/app/audios/extracted_audios/SVR01 \
  /workspace/app/test_one_patient/SVR01
ls -la /workspace/app/test_one_patient
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/test_one_patient \
  --output /workspace/app/depression-results/test/df_features.pkl \
  --feature-root /workspace/app/depression-results/test/features \
  --config configs/pipeline.yaml

---

###### 2 - How to test with only one feature - F0 ######
cp configs/pipeline.yaml configs/test_f0.yaml
nano configs/test_f0.yaml 
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/test_one_patient \
  --output /workspace/app/depression-results/test_f0.pkl \
  --feature-root /workspace/app/depression-results/test_f0 \
  --config configs/test_f0.yaml 

---

###### 3 - How to test only the transcription ######
cp configs/pipeline.yaml configs/test_transcription.yaml
transcription:
  enabled: true
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/test_one_patient \
  --output /workspace/app/depression-results/test_transcription.pkl \
  --feature-root /workspace/app/depression-results/test_transcription_features \
  --transcription-root /workspace/app/depression-results/test_transcriptions \
  --config configs/test_transcription.yaml

---

###### 4 - How to run the diarization validation ######
depression-validate-diarization \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/audios/extracted_audios \
  --validation-root /workspace/app/depression-nlp/data/diarization_validation \
  --config /workspace/app/depression-nlp/configs/pipeline.yaml


