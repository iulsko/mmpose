from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

def calculate_thr_cutoff(df, no_keypoints):
    cuts = np.zeros((no_keypoints,))
    for idx in range(no_keypoints):
        thresholds = df.iloc[idx::no_keypoints].kpt_threshold.to_numpy()
        nonan_threshold = thresholds[~np.isnan(thresholds)]
        q75, q25 = np.percentile(nonan_threshold, [75, 25])
        iqr = q75 - q25
        cut = q25 - 1.5*iqr # lower bound
        cuts[idx] = cut
    return cuts

def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuro:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.004, beta=0.7, d_cutoff=1.0):
        super(OneEuro, self).__init__()
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, x, t=None):
        """Compute the filtered signal."""
        if t is None:
            # Assume input is feed frame by frame if not specified
            t = self.t_prev + 1
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)  # [k, c]
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat) #f_c
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        
        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', help='path to cvs file with 2D detections')
    args = parser.parse_args()
    path = Path(args.path)
    
    # we use COCO dataset
    keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                      'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                      'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    n_keypoints = len(keypoint_names)

    df_2d_keypoints = pd.read_csv(path, index_col=0)
    df = df_2d_keypoints[["frame", "bbox_area"]]
    # in multiple detections pick ones with largest bbox area
    on_max = df.groupby(["frame"]).bbox_area.max().reset_index()
    df_2d_keypoints = pd.merge(df_2d_keypoints, on_max, on=["frame", "bbox_area"], how="inner")

    n_frames = max(df_2d_keypoints.frame) + 1

    df_2d_keypoints["xcoor_1euro"] = df_2d_keypoints["xcoor"]
    df_2d_keypoints["ycoor_1euro"] = df_2d_keypoints["ycoor"]
    df_2d_keypoints["kpt_threshold_1euro"] = df_2d_keypoints["kpt_threshold"]

    ####### smooth out x & y coordinates #######
    pose_result = np.zeros((n_frames, n_keypoints, 2))
    for idx in range(n_frames):
        res = df_2d_keypoints.iloc[n_keypoints * idx:n_keypoints * (idx + 1)]
        pose_result[idx] = res[["xcoor", "ycoor"]].values

    set_missing = set(df_2d_keypoints[df_2d_keypoints["bbox_area"] == -1].frame)
    set_next = set([el + 1 for el in set_missing])
    start_over = set_next - set_missing

    pose_result_filtered = np.zeros_like(pose_result)
    for i in range(0, len(pose_result)):
        if i in {0}|start_over:
            one_euro_filter = OneEuro(np.zeros_like(pose_result[i]),
                                      pose_result[i], 
                                      min_cutoff=0.1, beta=0.00001, d_cutoff=1.0)
            pose_result_filtered[i] = pose_result[i]
        elif i in set_missing:
            pose_result_filtered[i] = np.zeros((n_keypoints, 2))
        else:
            pose_result_filtered[i] = one_euro_filter(pose_result[i])

    ####### smooth out keypoint thresholds #######
    thr_cutoff = calculate_thr_cutoff(df_2d_keypoints, n_keypoints)
    for idx in range(n_keypoints):
        s = df_2d_keypoints.iloc[idx::17].kpt_threshold < thr_cutoff[idx]
        df_2d_keypoints.loc[s[s].index.values, "kpt_threshold_1euro"] = np.nan

    all_thresholds = np.zeros((n_frames, n_keypoints))
    list_of_start_overs, list_of_missings = [], []
    for i in range(n_keypoints):
        threshold = df_2d_keypoints.loc[i::n_keypoints].kpt_threshold_1euro.to_numpy()
        all_thresholds[:, i] = threshold
        set_missing = set(np.argwhere(np.isnan(threshold)).flatten())
        set_next = set(np.argwhere(np.isnan(threshold)).flatten() + 1)
        start_over = set_next - set_missing
        list_of_start_overs.append(start_over)
        list_of_missings.append(set_missing)
    
    pose_result_filtered_threshold = np.zeros_like(all_thresholds)
    for idx in range(n_keypoints):
        start_over = list_of_start_overs[idx]
        set_missing = list_of_missings[idx]
        for j in range(len(all_thresholds[:, idx])):
            if j in {0}|start_over:
                one_euro_filter = OneEuro(0,
                                          all_thresholds[j, idx], 
                                          min_cutoff=0.1, beta=0.00001, d_cutoff=1.0)
                pose_result_filtered_threshold[j, idx] = all_thresholds[j, idx]
            elif j in set_missing:
                pose_result_filtered_threshold[j, idx] = np.nan
            else:
                pose_result_filtered_threshold[j, idx] = one_euro_filter(all_thresholds[j, idx])

    for idx in (range(len(pose_result_filtered))):
        df_2d_keypoints.loc[n_keypoints * idx:n_keypoints * (idx + 1)-1, "xcoor_1euro"] = pose_result_filtered[idx][:,0]
        df_2d_keypoints.loc[n_keypoints * idx:n_keypoints * (idx + 1)-1, "ycoor_1euro"] = pose_result_filtered[idx][:,1]
        df_2d_keypoints.loc[n_keypoints * idx:n_keypoints * (idx + 1)-1, "kpt_threshold_1euro"] = pose_result_filtered_threshold[idx]
    df_2d_keypoints.to_csv(f"{path.parent}/1euro_{path.stem}.csv")


if __name__ == '__main__':
    main()
