# depression-nlp

Production-oriented acoustic pipeline for SDR.

## Current pipeline

1. Load the patient/df_ref spreadsheet.
2. Run local speaker diarization with `pyannote/speaker-diarization-community-1`.
3. Use the manually annotated patient interval to select the real patient embedding.
4. Match that embedding to the speakers found in every session.
5. Extract patient and therapist waveforms from exclusive diarization turns.
6. Extract F0, RMS energy, and 3D LogMel features from five-second chunks.
7. Store F0/energy values and LogMel `.npy` paths in the output DataFrame.

KMeans and MFCC are not used.

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

###### 1 -  How to test with only one patient - SVR01 ######
# Create a test folder:
AUDIO_ROOT=/workspace/app/audios/test_one_patient
mkdir -p /workspace/app/test_one_patient
# Check
ln -s \
  /workspace/app/audios/extracted_audios/SVR01 \
  /workspace/app/test_one_patient/SVR01
ls -la /workspace/app/test_one_patient
# Run 
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/test_one_patient \
  --output /workspace/app/depression-results/test/df_features.pkl \
  --feature-root /workspace/app/depression-results/test/features \
  --config configs/pipeline.yaml

###### 2 - How to test with only one feature - F0 ######
# Create a copy
cp configs/pipeline.yaml configs/test_f0.yaml
# Enable only F0
nano configs/test_f0.yaml 
# Run
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/test_one_patient \
  --output /workspace/app/depression-results/test_f0.pkl \
  --feature-root /workspace/app/depression-results/test_f0 \
  --config configs/test_f0.yaml 

###### 3 - How to test only the transcription ######
cp configs/pipeline.yaml configs/test_transcription.yaml
# Enable only transcription
transcription:
  enabled: true
# Run - one patient
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/test_one_patient \
  --output /workspace/app/depression-results/test_transcription.pkl \
  --feature-root /workspace/app/depression-results/test_transcription_features \
  --transcription-root /workspace/app/depression-results/test_transcriptions \
  --config configs/test_transcription.yaml

# Run - all patients
depression-features \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --audio-root /workspace/app/audios/extracted_audios \
  --output /workspace/app/depression-results/text_transcript.pkl \
  --feature-root /workspace/app/depression-results/transcription_features \
  --transcription-root /workspace/app/depression-results/transcriptions \
  --config configs/test_transcription.yaml

###### 4- How to run the diarization validation ######
python -m depression_nlp.diarization_validation.pipeline_validation

###### 5 - How to test the CBT screening ######
depression-cbt-screening \
  --input-root /workspace/app/depression-results/test_transcriptions \
  --output-root /workspace/app/depression-results/test_cbt_screening \
  --config configs/cbt_screening.yaml

###### 6 - Saving the CBT screening ######
python -m depression_nlp.deidentification.export_annotations \
  --input-root /workspace/app/depression-results/deidentified \
  --spreadsheet /workspace/app/planilhas/df_ref.xlsx \
  --output /workspace/app/depression-results/annotation_candidates.xlsx
