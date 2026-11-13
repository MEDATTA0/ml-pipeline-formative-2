# ml-pipeline-formative-2

Team: **Beulla**, **Ibrahim**, **Kelvin**

## Voice (Beulla)

- **Colab:** `notebooks/audio_voice_colab.ipynb` — upload the 6 WAV files from `data/audio/raw/`, run all, download `audio_features.csv`.
- **Deliverable:** `audio_features.csv` (repo root) + raw recordings in `data/audio/raw/`.
- **CLI scripts:** `src/formative_2/voice/` — see `docs/VOICE_CLI.md`.

## Data (team)

- `customer_social_profiles - customer_social_profiles.csv`
- `customer_transactions - customer_transactions.csv`


## Face and product CLI

From the project root, run these commands in your Python environment. Install
`torch torchvision` for the face command and `pandas joblib scikit-learn==1.6.1`
for the saved product model.

```bash
export PYTHONPATH=src
python -m formative_2.face.predict_face "data/images/Kelvin Aaron-Onuigbo.jpg.jpeg" --crop .15 .11 .87 .88
python -m formative_2.product.predict_product 151
```

Pass a cropped face or use `--crop` with the left, top, right, and bottom
fractions of the image. The face command reports the highest-scoring enrolled
member for the classroom demo. The product command builds features from
`customer_social_profiles.csv` for the supplied numeric customer ID.

## Transaction demo

From the project root, install the runtime dependencies and run the three models
together. Use scikit-learn 1.9.0 for the saved voice model. Loading the older
product model may print a scikit-learn version warning.

```bash
python -m pip install torch torchvision librosa soundfile pandas joblib scikit-learn==1.9.0
export PYTHONPATH=src

face="data/images/Kelvin Aaron-Onuigbo.jpg.jpeg"
crop=(.15 .11 .87 .88)

# Approved transaction
python -m formative_2 kelvin "$face" data/audio/raw/kelvin_yes_approve.wav --crop "${crop[@]}"

# Denied at the face check
python -m formative_2 beulla "$face" data/audio/raw/beulla_yes_approve.wav --crop "${crop[@]}"

# Denied at the voice check
python -m formative_2 kelvin "$face" data/audio/raw/beulla_yes_approve.wav --crop "${crop[@]}"
```

The demo links Kelvin, Beulla, and Ibrahim to sample customer IDs 151, 192,
and 114. It checks the claimed account against the face prediction, computes
the product category, then checks the same account against the voice prediction.
It displays the category only after both checks pass. The sample recordings
demonstrate speaker matching. The voice model does not transcribe the phrase.
The demo uses classroom models and uncalibrated confidence thresholds.
