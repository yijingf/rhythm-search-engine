import re
import os
import librosa
import pretty_midi
import numpy as np
import pandas as pd

from glob import glob


ver = ["monophonic", "polyphonic"]
csv_header = ["ontime/beat", "note",
              "morephetic pitch", "duration/beat", "staff number"]


# meter_re = re.compile(r'\*M\d\/\d')
bpm_re = re.compile(r'\*MM\d+')


def get_bpm(fname):
    """Get BPM from .krn file

    Args:
        fname(_type_): _description_
    """
    with open(fname, "r") as f:
        data = f.read().splitlines()
    strs = [i for i in data if '*M' in i]

    strs = " ".join(strs)

#     meter_str = meter_re.findall(strs)
#     meter = meter_str[0]

    bpm_str = bpm_re.findall(strs)

    if not len(bpm_str):
        assert f"unknown tempo for {fname}"
        return

    bpm = re.findall(r"\d+", bpm_str[0])[0]

    return int(bpm)


def get_audio_info(audio_fname):
    """Get audio onset time of first note, and audio duration in sec.

    Args:
        audio_fname (_type_): _description_

    Returns:
        _type_: _description_
    """
    s, fs = librosa.load(audio_fname)
    idx = np.where(np.diff(np.where(s == 0)[0]) != 1)[0]
    if not len(idx):
        idx = len(np.where(s == 0)[0])
    else:
        idx = idx[0]
    return idx / fs, len(s) / fs


def integrate_metadata(data_dir):
    """_summary_

    Returns:
        _type_: _description_
    """
    df = pd.DataFrame(columns=["piece", "version",
                               "bpm",
                               "midi start/s",
                               "midi end/s",
                               "audio start/s",
                               "audio end/s",
                               "beat start",
                               "beat end"])

    cnt = 0

    pieces = [i for i in os.listdir(data_dir) if i[0] != "."]

    for v in ver:

        for p in pieces:
            p_dir = os.path.join(data_dir, p, v)

            try:
                krn_fname = glob(os.path.join(p_dir, "kern", "*.krn"))[0]
                bpm = get_bpm(krn_fname)
            except:
                print(f".krn file not found in {p_dir}")
                continue

            try:
                midi_fname = glob(os.path.join(p_dir, "midi", "*.mid"))[0]
                pm = pretty_midi.PrettyMIDI(midi_fname)
                midi_st = pm.instruments[0].notes[0].start
                midi_ed = pm.get_end_time()

            except:
                print(f".mid file not found in {p_dir}")
                continue

            try:
                csv_fname = glob(os.path.join(p_dir, "csv", "*.csv"))[0]
                df = pd.read_csv(csv_fname, header=None, names=csv_header)
                beat_st = df['ontime/beat'].iloc[0]
                last_beat = df['ontime/beat'].iloc[-1]
                beat_ed = max(df.loc[df['ontime/beat'] ==
                              last_beat, 'duration/beat']) + last_beat
            except:
                print(f".csv file not found in {p_dir}")
                continue

            try:
                audio_fname = glob(os.path.join(p_dir, "audio", "*.wav"))[0]
                audio_st, audio_ed = get_audio_info(audio_fname)
            except:
                print(f".wav file not found in {p_dir}")

            df.loc[cnt] = {"piece": p, "version": v, "bpm": bpm,
                           "midi start/s": midi_st, "midi duration/s": midi_ed,
                           "audio start/s": audio_st, "audio duration/s": audio_ed,
                           "beat start": beat_st, "beat end": beat_ed}

    return df


def get_child_dirs(dirname):
    child_dirs = [i for i in os.listdir(dirname)
                  if os.path.isdir(os.path.join(dirname, i))]
    return child_dirs


def get_pattern_info(data_dir):

    df = []

    pieces = [i for i in os.listdir(data_dir) if i[0] != "."]

    for v in ver:

        for p in pieces:
            p_dir = os.path.join(data_dir, p, v, "repeatedPatterns")
            tasks = get_child_dirs(p_dir)

            if "barlowAndMorgensternRevised" in tasks:
                try:
                    tasks.remove("barlowAndMorgenstern")
                except:
                    continue

            pattern_df = []
            for t in tasks:
                t_dir = os.path.join(p_dir, t)
                patterns = get_child_dirs(t_dir)

                for pattern in patterns:
                    occs = glob(os.path.join(t_dir, pattern,
                                "occurrences", "csv", "*.csv"))
                    interval = set()
                    for occ_fname in occs:
                        occ_df = pd.read_csv(occ_fname, header=None)
                        interval.add((occ_df.iloc[0][0], occ_df.iloc[-1][0]))

                    integrated_df = pd.DataFrame(
                        interval, columns=["start/beat", "end/beat"])
                    integrated_df["pattern"] = pattern
                    pattern_df.append(integrated_df)

            pattern_df = pd.concat(pattern_df)
            pattern_df["piece"] = p
            pattern_df["version"] = v

            df.append(pattern_df)

    df = pd.concat(df)
    return df


if __name__ == "__main__":

    data_dir = "../data/JKUPDD-Aug2013/groundTruth"

    # Gather metadata from .krn file, midi and audio
    meta_df = integrate_metadata(data_dir)

    # Manually correct bpm based on the beat and audio duration
    meta_df.loc[(meta_df["piece"] == "beethovenOp2No1Mvt3") &
                (meta_df["version"] == "monophonic"), "bpm"] = 192
    meta_df.loc[(meta_df["piece"] == "gibbonsSilverSwan1612") &
                (meta_df["version"] == "monophonic"), "bpm"] = 150

    meta_df.to_csv("../metadata/mirex.csv", index=False)

    # Get the start/ending beats of all patterns
    pattern_df = get_pattern_info(data_dir)

    # Convert timestamps from beat to second
    pattern_df = pd.merge(pattern_df, meta_df, on=['piece', 'version'])
    pattern_df['start/s'] = (pattern_df['start/beat'] -
                             pattern_df['beat start']) * 60 / pattern_df['bpm']
    pattern_df['end/s'] = (pattern_df['end/beat'] -
                           pattern_df['beat start']) * 60 / pattern_df['bpm']

    pattern_df = pattern_df[['piece', 'version', 'start/beat', 'end/beat',
                             'pattern', 'start/s', 'end/s']]
