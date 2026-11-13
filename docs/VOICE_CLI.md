# Voice CLI (Beulla only)

Does not modify face or product code. Teammates call these from the shared CLI when ready.

```bash
export PYTHONPATH=src
pip install librosa soundfile pandas scikit-learn joblib matplotlib

python -m formative_2.voice.process_audio      # optional
python -m formative_2.voice.train_voice_model  # writes models/voice_verifier.joblib
python -m formative_2.voice.verify_voice data/audio/raw/beulla_yes_approve.wav
```

`verify_voice` exits **0** if approved, **1** if denied.

Colab workflow: `notebooks/audio_voice_colab.ipynb` (primary).
