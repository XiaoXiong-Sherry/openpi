from vlaholo.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import os
import shutil
import sys
import sys

"""
1. filtering right single arm action data only

"""


def resave_lerobot_to_single_arm(data_path):

    ds = LeRobotDataset(repo_id=data_path, root=data_path)
   

    save_root = data_path.rstrip("/") + "_novideo_data"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)

    # use_video = False
    use_video = False

    dataset_new = LeRobotDataset.create(
        repo_id="data_new/test_aloha_new",
        robot_type="qiuzhi",
        fps=20,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    # "right_waist",
                    # "right_shoulder",
                    # "right_elbow",
                    # "right_forearm_roll",
                    # "right_wrist_angle",
                    # "right_wrist_rotate",
                    # "right_gripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    # "right_waist",
                    # "right_shoulder",
                    # "right_elbow",
                    # "right_forearm_roll",
                    # "right_wrist_angle",
                    # "right_wrist_rotate",
                    # "right_gripper",
                ],
            },
            "observation.images.cam_high": {
                "dtype": "video" if use_video else "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "video" if use_video else "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
        
        },
        root=save_root,
        use_videos=use_video,
    )

    ds_episodes_index = ds.episode_data_index
    # from id to id
    print(ds_episodes_index)
    # {'from': tensor([   0,  113,  223,  344,  466,  582,  694,  821,  934, 1049]), 'to': tensor([ 113,  223,  344,  466,  582,  694,  821,  934, 1049, 1162])}
    from_indices = ds_episodes_index["from"].cpu().numpy()
    to_indices = ds_episodes_index["to"].cpu().numpy()
    import numpy as np

    for ep_idx in range(len(from_indices)):
        start_idx = int(from_indices[ep_idx])
        end_idx = int(to_indices[ep_idx])
        for frame_idx in range(start_idx, end_idx):
            itm = ds[frame_idx]
            dataset_new.add_frame(
                {
                    "observation.state": itm["observation.state"],
                    "action": itm["action"],
                    "observation.images.cam_high": itm["observation.images.cam_high"],
                    "observation.images.cam_left_wrist": itm[
                        "observation.images.cam_left_wrist"
                    ],
                },
                task=itm["task"],
            )
        dataset_new.save_episode()
    
    # dataset_new.save_episode()
    print(f"Total episodes saved: {dataset_new.__len__}")


data_path = sys.argv[1]
resave_lerobot_to_single_arm(data_path)
