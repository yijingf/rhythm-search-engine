"""Calculating the Dynamic Periodicity Warping (DPW) between two periodicty spectra as described in: 

Rhythmic similarity of music based on dynamic periodicity warping (https://ieeexplore.ieee.org/document/4518085) by Andre Holzapfel et al.


Returns:
    _type_: _description_
"""


import scipy
import scipy.sparse

import numpy as np
from numpy.linalg import norm

# Use DP to find lowest cost path through matrix D


def min_cost_path(s):
    m, n = s.shape
    cost = np.zeros_like(s) + np.nan

    cost[0, :] = np.cumsum(s[0, :])
    cost[:, 0] = np.cumsum(s[:, 0])
    pos = []

    for i in range(1, m):
        tmp_pos = []
        for j in range(1, n):
            candidate = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
            idx = np.argmin(candidate)
            tmp_pos.append([(i - 1, j - 1), (i - 1, j), (i, j - 1)][idx])
            cost[i, j] = candidate[idx] + s[i, j]
        pos.append(tmp_pos)

    pos = [[(-1, -1) for _ in range(n - 1)]] + pos
    new_pos = [[(-1, -1)] + i for i in pos]

    i, j = new_pos[-1][-1]
    path = [[i, j]]
    while i >= 0:
        i, j = new_pos[i][j]
        path.append([i, j])

    path = path[::-1]
    return cost, new_pos, np.array(path)

# Find the reference path by drawing a sub-diagonal line pass through the local minimum


def ref_path_offset(s, offset=2):
    offsets = (np.arange(-offset, offset + 1)).tolist()
    values = [np.diagonal(s, offset=i) for i in offsets]
    tridiagnoal_s = scipy.sparse.diags(values, offsets).toarray()

    x, y = np.where(tridiagnoal_s == np.max(tridiagnoal_s))
    coord = np.array([x[0], y[0]])

    # find the straight line passing through local maximum
    delta = y[0] - x[0]
    return delta


def DPW_distance(x, y, offset=2, theta=-np.pi / 4):
    # f_res: Frequency resolution of beat spectra, see ./src/compute_beat_spectra
    # theta: rotation angle for calculating the projection

    # difference matrix
    x1 = np.repeat(np.expand_dims(x, 1), len(x), axis=1)
    y1 = np.repeat(np.expand_dims(y, 1), len(y), axis=1).T

    d = np.power(x1 - y1, 2)

    # min cost path from d
    _, _, path = min_cost_path(d)

    # similarity matrix
    s = np.outer(x, y)

    # reference path offset
    rho_offset = ref_path_offset(s, offset)

    # projection
    # rotate paths by -45, and only take y value
    rot_vec = np.array([np.sin(theta), np.cos(theta)])
    rot_path_y = np.dot(rot_vec, path.T)

    rot_rho_y = rho_offset * np.cos(theta)

    avg_proj_dist = np.mean(np.abs(rot_path_y - rot_rho_y))
    return avg_proj_dist


if __name__ == "__main__":

    import time

    ref_p_f = dict(np.load("../data/beat_spectra/K448orig_120.npz"))['p_f']
    p_f = dict(np.load("../data/maestro_selected.npz",
               allow_pickle=True))['p_f']

    output_fname = "../data/dpw_score.npz"

    t0 = time.time()

    scores = []

    # Looping through all pieces
    for p_f_i in p_f:

        # normalize
        p_f_i = np.array(p_f_i.T)
        p_f_i = (p_f_i / norm(p_f_i, axis=0)).T

        score_i = []

        # Looping through all segments of one piece
        for win in p_f_i:
            score_win = []

            # Looping through all boundaries
            for ref_win in ref_p_f:
                score_win.append(DPW_distance(ref_win, win))

            score_i.append(score_win)
        scores.append(score_i)
        
    scores = np.array(scores, dtype=object)
    np.savez_compressed(output_fname, **{"score": scores}, allow_pickle=True)

    print(time.time() - t0)
