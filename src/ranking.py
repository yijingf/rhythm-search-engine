import numpy as np
import pandas as pd


def sec2min(t):
    m, s = t // 60, t % 60
    return f"00:{m:02d}:{s:02d}"


def sort_max(max_scores):
    """Sorted by max similarity.

    Args:
        max_scores (_type_): _description_

    Returns:
        _type_: _description_
    """

    max_scores_idx = []
    for song_idx, score in enumerate(max_scores):

        for win_idx, win in enumerate(score):

            # add index for retrieval
            max_scores_idx.append((song_idx, win_idx, win))

    sorted_max_scores = sorted(max_scores_idx,
                               key=lambda x: x[-1], reverse=True)

    return sorted_max_scores


def sort_all(scores):
    index_scores = []
    for song_id, score in enumerate(scores):
        for win_id, win_score in enumerate(score):
            for phrase_id, phrase_score in enumerate(win_score):
                index_scores.append((song_id, win_id, phrase_score, phrase_id))

    sorted_scores = sorted(index_scores, key=lambda x: x[-2], reverse=True)
    return sorted_scores


def print_top_k_seg(sorted_scores, df, top_k=10, t_hop=5):
    top_k_df = pd.DataFrame({"composer": [],
                            "title": [],
                             "audio_filename": [],
                             "start": []})

    for i in range(top_k):
        song_id, win_id, score = sorted_scores[i][:3]
        row = df.loc[song_id]

        top_k_df.loc[i] = {"composer": row['canonical_composer'],
                           "title": row['canonical_title'],
                           "audio_filename": row['audio_filename'],
                           "start": f"{sec2min(win_id*t_hop)}"}

    return top_k_df


def merge_index(idx):
    if not len(idx):
        return []

    st_ed = []

    st = idx[0]
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > 1:
            if idx[i - 1] == st:
                st = idx[i]
                continue
            else:
                st_ed.append((st, idx[i - 1]))
                st = idx[i]
        else:
            continue

    if st != idx[i]:
        st_ed.append((st, idx[i]))

    return st_ed


def format_interval(st_ed_idx, t_hop=5, t_frame=10):

    intervals = []
    for st_idx, ed_idx in st_ed_idx:
        t_st = sec2min(st_idx * t_hop)
        t_ed = sec2min(ed_idx * t_hop + t_frame)
        intervals.append(f"{t_st}~{t_ed}")
    return intervals


def print_top_k_piece(sorted_score, df, top_k=10):

    top_k_df = pd.DataFrame({"composer": [],
                             "title": [],
                             "audio_filename": [],
                             "interval": []})

    for i in range(top_k):
        song_id, segment_idx = sorted_score[i]
        row = df.iloc[sorted_score[i][0]]

        st_ed_idx = merge_index(segment_idx)
        intervals = format_interval(st_ed_idx)

        top_k_df.loc[i] = {"composer": row['canonical_composer'],
                           "title": row['canonical_title'],
                           "audio_filename": row['audio_filename'],
                           "interval": set(intervals)}

    return top_k_df


def temporal_cluster_ratio(s_i, thresh=0.8, cluster_len=3):
    idx = np.where(s_i > thresh)[0]
    if not len(idx):
        return 0

    total_cnt = 0
    cnt = 0

    last = idx[0]
    for i, curr in enumerate(idx[1:]):
        cnt += 1
        if curr - idx[i] > 1:
            if cnt < cluster_len:
                cnt = 0
                continue
            else:

                total_cnt += cnt
                cnt = 0
        else:
            continue

    return total_cnt / len(s_i)
