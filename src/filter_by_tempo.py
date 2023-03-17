"""Search for songs with tempo between 120 and 144 BPM, return a `selected_maestro.csv`.
Estimate the tempo of 3 segments of 30-s extracted from the beginning, middle and end of each song.
"""

import os
import librosa
import numpy as np
import pandas as pd


max_tempo = 144
min_tempo = 120


def estimate_tempo(fname, duration, t_seg=30):
    tempo = []

    # Estimate the tempo of 3 segments extracted from the beginning, middle and end of the song
    for st in [0, duration - t_seg / 2, duration - t_seg]:
        y, sr = librosa.load(fname, offset=st, duration=t_seg)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo.append(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))

    return tempo


if __name__ == "__main__":

    root_dir = "/isi/music/yijing/maestro-v3.0.0/"
    df = pd.read_csv(os.path.join(root_dir, "maestro-v3.0.0.csv"))

    tempi = []
    for _, row in df.iterrows():
        duration = row['duration']
        fname = os.path.join(root_dir, 'audio', row['audio_filename'])
        tempi.append(estimate_tempo(fname, duration))

    tempi = np.squeeze(np.array(tempi))
    avg_tempi = np.mean(tempi, axis=1)

    selected = np.logical_and(np.min(tempi, axis=1) <= max_tempo,
                              np.max(tempi, axis=1) >= min_tempo)
    selected = selected & (avg_tempi >= min_tempo) & (avg_tempi <= max_tempo)
    selected_df = df[selected].reset_index(drop=True)

    os.makedirs("../metadata", exist_ok=True)
    np.save("../metadata/maestro_tempi.npy", tempi)
    selected_df.to_csv("../metadata/maestro_selected.csv", index=False)
