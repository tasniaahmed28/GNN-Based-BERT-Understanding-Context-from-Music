"""Audio loading and feature extraction: mel-spectrograms and chroma segmentation."""
import numpy as np
import librosa

SR = 22050
SEGMENT_SEC = 5.0
MAX_DURATION = 30.0       # FMA-small clips are ~30s
MC_SEGMENT_SEC = 2.0      # MusicCaps clips are ~10s -> 2s segments = 5 nodes
N_MELS = 64
FIXED_FRAMES = 130


def track_id_to_path(track_id, fma_dir="/content/gnn-bert-music-context/data/raw/fma_small"):
    """Map an FMA track_id to its mp3 path (FMA's zero-padded 3-level folder scheme)."""
    tid_str = f"{track_id:06d}"
    return f"{fma_dir}/{tid_str[:3]}/{tid_str}.mp3"


def extract_segment_chroma(path, sr=SR, segment_sec=SEGMENT_SEC, max_duration=MAX_DURATION):
    """Split an FMA track into fixed windows and extract mean chroma per segment.
    Returns array of shape (n_segments, 12), or None if too short / unreadable."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=max_duration)
    except Exception:
        return None
    if len(y) < sr:
        return None

    seg_len = int(segment_sec * sr)
    n_segments = len(y) // seg_len
    if n_segments < 2:  # need at least 2 nodes for a graph
        return None

    segment_features = []
    for i in range(n_segments):
        seg = y[i * seg_len: (i + 1) * seg_len]
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
        segment_features.append(chroma.mean(axis=1))
    return np.stack(segment_features)


def extract_segment_chroma_mc(path, sr=SR, segment_sec=MC_SEGMENT_SEC):
    """Same as extract_segment_chroma, tuned for shorter MusicCaps clips."""
    try:
        y, _ = librosa.load(path, sr=sr)
    except Exception:
        return None
    if len(y) < sr:
        return None

    seg_len = int(segment_sec * sr)
    n_segments = len(y) // seg_len
    if n_segments < 2:
        return None

    segment_features = []
    for i in range(n_segments):
        seg = y[i * seg_len: (i + 1) * seg_len]
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
        segment_features.append(chroma.mean(axis=1))
    return np.stack(segment_features)


def extract_melspec(path, sr=SR, n_mels=N_MELS, fixed_frames=FIXED_FRAMES, duration=MAX_DURATION):
    """Fixed-size log-mel spectrogram for the CNN baseline. Shape: (n_mels, fixed_frames)."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)
    except Exception:
        return None
    if len(y) < sr:
        return None

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :fixed_frames]
    return mel_db
