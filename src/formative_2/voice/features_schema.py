"""Feature column names (no librosa dependency — train CLI can run with CSV only)."""

N_MFCC = 13


def feature_column_names() -> list[str]:
    cols = [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    cols += [f"mfcc_{i}_std" for i in range(N_MFCC)]
    cols += [
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "energy_mean",
        "energy_std",
        "duration_sec",
    ]
    return cols
