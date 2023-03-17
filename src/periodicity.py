"""Extract periodicity features from audio.
"""

import librosa
import numpy as np
import pandas as pd
from scipy.fft import fft


def periodicity_spectrum(y=None, onset_env=None, sr=22050, max_bpm=1000, min_bpm=10):

    if onset_env is None:
        D = np.abs(librosa.feature.melspectrogram(y=y, sr=sr))
        onset_env = librosa.onset.onset_strength(S=librosa.core.power_to_db(D),
                                                 sr=sr)
    t_frame = librosa.times_like(onset_env, sr=sr)[-1]
    frame_len = len(onset_env)

    spectrum = np.abs(fft(onset_env))
    spectrum = spectrum[1:frame_len // 2]
    spectrum = spectrum / np.sum(spectrum)

    max_idx = int(max_bpm / 60 * t_frame)
    min_idx = int(min_bpm / 60 * t_frame)

    spectrum = spectrum[min_idx: max_idx]
    return spectrum


def periodicity_spectrogram(y, sr=22050, t_frame=10, t_hop=5, max_bpm=1000, min_bpm=10):
    """Calculate periodicity spectra using method described in:

    Rhythmic similarity of music based on dynamic periodicity warping (https://ieeexplore.ieee.org/document/4518085) by A. Holzapfel et al.

    The code differs from the paper in the way of calculating onset strength.

    Frequency resolution: 1/t_frame

    Returns:
        _type_: _description_
    """

    D = np.abs(librosa.feature.melspectrogram(y=y, sr=sr))
    onset_env = librosa.onset.onset_strength(S=librosa.core.power_to_db(D),
                                             sr=sr)

    # For calculating frequency resolution
    times = librosa.times_like(D, sr=sr)
    onset_env_sr = np.mean(1 / np.diff(times))

    frame_len = int(t_frame * onset_env_sr)

    spectrogram = []
    t = len(y) / sr
    for t_st in np.arange(0, t - t_frame, t_hop):
        idx_st = int(t_st * onset_env_sr)
        spectrogram.append(fft(onset_env[idx_st:idx_st + frame_len]))

    spectrogram = np.array(np.abs(spectrogram)).T

    spectrogram = spectrogram[1:frame_len // 2, :]
    spectrogram = spectrogram / np.sum(spectrogram, axis=0)  # normalize

    max_idx = int(max_bpm / 60 * t_frame)
    min_idx = int(min_bpm / 60 * t_frame)

    return spectrogram[min_idx:max_idx, :]


def beat_spectrum(y, sr=22050, frame_len=256, hop_len=128, method='baseline'):
    """
    Calculating beat spectra using method described in:

    The beat spectrum: a new approach to rhythm analysis (https://ieeexplore.ieee.org/document/1237863/) by J. Foote et al.
    """

    S = np.abs(librosa.stft(y, n_fft=frame_len,
               hop_length=hop_len, win_length=frame_len))
    S_db = librosa.power_to_db(S**2, ref=1)
    norm_S_db = S_db / np.sqrt(np.sum(np.square(S_db), axis=0))

    # Cosine similarity matrix of S
    s = np.dot(norm_S_db.T, norm_S_db)

    t_lag = int(len(y) / sr / 2)
    n = int(t_lag / (1 / sr * hop_len))

    if method == 'baseline':
        # Summing along the diagonal
        bs = [np.trace(s[i:i + n, :n]) for i in range(n)]
    else:
        # Todo: Auto-correlation
        bs = []

    return bs


def beat_spectrogram(y, sr=22050, t_frame=10, t_hop=5, win_len=256, hop_len=128):
    t = len(y) / sr

    spectrogram = []
    for i in np.arange(0, int((t - t_frame) / t_hop)):
        t_st = i * t_hop
        t_ed = i * t_hop + t_frame

        y_win = y[int(t_st * sr): int(t_ed * sr)]
        bs = beat_spectrum(y_win, win_len=win_len, hop_len=hop_len)
        spectrogram.append(bs)

    spectrogram = np.array(spectrogram)
    spectrogram = spectrogram.T[::-1, :] / np.max(spectrogram)

    return spectrogram


if __name__ == "__main__":

    import os
    import time
    import argparse

    df_fname = "../metadata/maestro_selected.csv"
    audio_dir = "/isi/music/yijing/maestro-v3.0.0/audio"
    output_fname = "../data/periodicity_spectra/maestro_selected.npz"

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="fname",
                        default=df_fname, type=str, help="Path to .csv file of music.")
    parser.add_argument("-d", dest="audio_dir",
                        default=audio_dir, type=str, help="Parent directory to all audio.")
    parser.add_argument("-o", dest="output_fname",
                        default=output_fname, type=str, help="File name of output .csv")
    parser.add_argument("-m", dest="method",
                        default="periodicity", type=str, help="Method of periodicty estimation.")

    args = parser.parse_args()

    df = pd.read_csv(args.fname)

    t0 = time.time()

    spectrograms = []
    for _, row in df.iterrows():
        fname = os.path.join(args.audio_dir, row['audio_filename'])
        y, sr = librosa.load(fname)

        if args.method == 'periodicity':
            spectrogram = periodicity_spectrogram(y)
        else:
            spectrogram = beat_spectrogram(y)

        spectrograms.append(spectrogram.T)

    spectrograms = np.array(spectrograms, dtype=object)
    np.savez_compressed(args.output_fname, **
                        {"spectrogram": spectrograms}, allow_pickle=True)

    t = (time.time() - t0) / 60
    print(f"Finished in {t:.2f} min.")
