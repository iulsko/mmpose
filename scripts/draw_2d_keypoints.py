from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# this is a COCO dataset
# https://github.com/open-mmlab/mmpose/blob/eeebc652842a9724259ed345c00112641d8ee06d/mmpose/apis/inference.py#L540

# palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
#                     [230, 230, 0], [255, 153, 255], [153, 204, 255],
#                     [255, 102, 255], [255, 51, 255], [102, 178, 255],
#                     [51, 153, 255], [255, 153, 153], [255, 102, 102],
#                     [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                     [51, 255, 51], [0, 255, 0], [0, 0, 255],
#                     [255, 0, 0], [255, 255, 255]])
# skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
#             [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
#             [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
#             [3, 5], [4, 6]]
# pose_link_color = palette[[
#     0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
# ]]
# pose_kpt_color = palette[[
#     16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
# ]]

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255]])
skeleton = [[13, 11], [14, 12], [11, 12], [5, 11], [6, 12], 
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], 
            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
pose_link_color = palette[[
    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16
]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0
]]
name_keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
                  'right_knee', 'left_ankle', 'right_ankle']

# https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/visualization/image.py
def draw_2d_skeleton(kpts, frame):
    img_h, img_w, _ = frame.shape
    radius = 5
    kpt_score_thr = 0
    thickness = 2
    kpts = np.delete(kpts, [15, 16], axis=0)
    # draw each point on image
    assert len(pose_kpt_color) == len(kpts)

    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
        if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
            # skip the point that should not be drawn
            continue
        color = tuple(int(c) for c in pose_kpt_color[kid])
        cv2.circle(frame, (int(x_coord), int(y_coord)), radius, color, -1)
        cv2.putText(frame, name_keypoints[kid], (int(x_coord-10), int(y_coord-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # draw links
    assert len(pose_link_color) == len(skeleton)
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0 
            or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
            or pos2[1] <= 0 or pos2[1] >= img_h
            or kpts[sk[0], 2] < kpt_score_thr
            or kpts[sk[1], 2] < kpt_score_thr
            or pose_link_color[sk_id] is None):
            # skip the link that should not be drawn
            continue
        color = tuple(int(c) for c in pose_link_color[sk_id])
        cv2.line(frame, pos1, pos2, color, thickness=thickness)
    return frame

def main():
    parser = ArgumentParser()
    parser.add_argument('--path_mp4', help='where mp4s are stored', required=True)
    parser.add_argument('--path_csv', help='where csvs for corresponding mp4s are stored', required=True)
    parser.add_argument('--single', help='only consider one mp4 and csv file', required=False, action='store_true')
    parser.add_argument('--smooth', help='where to use smoothed 1euro values', required=False, action='store_true')
    args = parser.parse_args()
    path_mp4 = Path(args.path_mp4)
    path_csv = Path(args.path_csv)
    single = args.single
    smooth = args.smooth

    if not single:
        path_mp4 = path_mp4.glob("*.mp4")
    else:
        path_mp4 = [path_mp4]

    if smooth:
        columns = ["xcoor_1euro", "ycoor_1euro", "kpt_threshold_1euro"]
        smth = "smooth_"
    else:
        columns = ["xcoor", "ycoor", "kpt_threshold"]
        smth = ""

    for video_path in path_mp4:
        if not single:
            possible_csvs = path_csv.glob(f"beta*{video_path.stem}.csv")
        else:
            possible_csvs = [path_csv]

        for possible_csv in possible_csvs:
            print("processing", possible_csv, "\n")
            result = pd.read_csv(possible_csv, index_col=0)
            video = cv2.VideoCapture(video_path.as_posix())
            
            assert len(set(result.frame)) == int(video.get(cv2.CAP_PROP_FRAME_COUNT)), "csv and mp4 don't match"
            n_frames = result.shape[0] // len(set(result.keypoint_name))
            n_keypoints = len(set(result.keypoint_name))
            unique_frames = list(dict.fromkeys(result.frame.tolist()))

            pose_result = []
            for idx in tqdm(range(n_frames)):
                res = result.iloc[n_keypoints * idx:n_keypoints * (idx + 1)]
                res_2d = res[columns].values
                pose_result.append(res_2d)

            fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
            fps = video.get(cv2.CAP_PROP_FPS)
            size_original = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            writer = cv2.VideoWriter((possible_csv.parents[0] / f"{smth}{possible_csv.stem}.mp4").as_posix(), fourcc, fps, size_original)

            i = 0
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # if not np.isnan(np.min(pose_result[i][:2])):
                #     # don't draw if non-detected keypoints
                frame = draw_2d_skeleton(pose_result[i], frame)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                i += 1
            video.release()
            writer.release()

if __name__ == '__main__':
    main()
