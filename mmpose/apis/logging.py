import pandas as pd
import numpy as np
import sys

def log_2d_results(pose_results, dataset_keypoints):
    """
    Log results of 2d detection demo to a csv file.
    Args:
        pose_results (list) : list of 2d detections obtained using inference_top_down_pose_model

    Returns:
        df_2d_keypoints (pandas.DataFrame) : dataframe containing 2D detections
    """
    dic_2d_keypoints = {"xcoor": [], "ycoor": [], "kpt_threshold": [], 
                       "bbox_threshold": [], "bbox_area": [], "bbox": [],
                       "person": [], "keypoint_name": [], "frame": []}

    for frame_id, pose_result in enumerate(pose_results):
        # a person was detected in the frame
        if pose_result:
            # there might be multiple people detected
            n_people = len(pose_result)
            poses = np.vstack([pose_result[i]["keypoints"] for i in range(n_people)])
            dic_2d_keypoints["xcoor"] += poses[:,0].tolist()
            dic_2d_keypoints["ycoor"] += poses[:,1].tolist()
            dic_2d_keypoints["kpt_threshold"] += poses[:,2].tolist()
            # bbox values (topleft - bottomright)
            bbox = [pose_result[i]["bbox"][:4] for i in range(n_people) for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["bbox"] += bbox
            # bbox threshold
            bbox_threshold = [pose_result[i]["bbox"][-1] for i in range(n_people) for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["bbox_threshold"] += bbox_threshold
            # bbox area
            bbox_area = [abs(pose_result[i]["bbox"][2] - pose_result[i]["bbox"][0]) * abs(pose_result[i]["bbox"][3] - pose_result[i]["bbox"][1]) \
                         for i in range(n_people) for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["bbox_area"] += bbox_area
            # keypoint name
            names = [dataset_keypoints[rep]["name"] for rep in range(len(dataset_keypoints)) for i in range(n_people)]
            dic_2d_keypoints["keypoint_name"] += names
            # # track id
            # trackids = [pose_result[i]["track_id"] for i in range(n_people) for rep in range(len(dataset_keypoints))]
            # dic_2d_keypoints["track_id"].append(trackids)
            # person id (might be multiple people)
            persons = [i for i in range(n_people) for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["person"] += persons
            frames = [frame_id for i in range(n_people) for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["frame"] += frames
        else:
            print("missing frame", frame_id)
            # fill in missing frames with nan
            nans = [np.nan for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["xcoor"] += nans
            dic_2d_keypoints["ycoor"] += nans
            dic_2d_keypoints["kpt_threshold"] += nans
            dic_2d_keypoints["bbox"] += nans
            dic_2d_keypoints["bbox_threshold"] += nans
            dic_2d_keypoints["bbox_area"] += nans
            names = [dataset_keypoints[rep]["name"] for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["keypoint_name"] += names
            dic_2d_keypoints["track_id"] += nans
            dic_2d_keypoints["person"] += nans
            frames = [frame_id for rep in range(len(dataset_keypoints))]
            dic_2d_keypoints["frame"] += frames          

    df_2d_keypoints = pd.DataFrame(dic_2d_keypoints)
    return df_2d_keypoints
