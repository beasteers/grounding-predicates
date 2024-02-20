import typing
import os
import re
import json
import random
import pickle
import pathlib
import tqdm
import h5py
import hdf5plugin
import concurrent

import symbolic
from env import twentybn
from gpred import video_utils, dnf_utils
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns






def load_something_else(paths):
    """
    sth_else = {
        "{id}": [
            {
                "name": "{id}/####.jpg",
                "labels": [
                    {
                        "box2d": {
                            "x1": float,
                            "x2": float,
                            "y1": float,
                            "y2": float,
                        },
                        "category": "battery",
                        "gt_annotation": "object 0",
                        "standard_category": "0000",
                    }
                ],
                "gt_placeholders": ["battery"],
                "nr_instances": 1},
            }
        ]
    }
    """
    sth_else = {}
    for i in tqdm.tqdm(range(4)):
        with open(os.path.join(paths.data,f'SomethingElse/bounding_box_smthsmth_part{i+1}.json'), "r") as f:
            for key, frames in json.load(f).items():
                sth_else[key] = frames
    return sth_else


def create_video_labels_dataset(paths, video_labels: typing.Dict, action_labels: typing.List, action_instances: typing.List):
    """Stores the Something-Else labels in h5py format.

    h5py = {
        "actions": {
            "id_action": {
                attrs: {
                    "id_action": int,
                    "template": utf8,
                },
                "videos": [V] (num_videos) uint32,
            }
        },
        "videos": {
            "id_video": {
                attrs: {
                    "id_video": int,
                    "id_action": int,
                },
                "objects": [O] (num_objects) utf8,
                "keyframes": [T] (num_keyframes) uint32,
                "boxes": [T, 1 + O, 4] (num_keyframes, hand/num_objects, x1/y1/x2/y2) float32,
            }
        }
    }
    
    Args:
        video_labels: Something-Else labels.
    """
    if os.path.isfile(paths.data / "labels.hdf5"):
        print(paths.data / "labels.hdf5", "exists")
        return
    with h5py.File(paths.data / "labels.hdf5", "w") as f:
        # Prepare action labels.
        dset_actions = f.create_group("actions")
        A = len(action_labels)
        for id_action in range(A):
            grp = dset_actions.create_group(str(id_action))
            grp.attrs.create("id_action", id_action, dtype=np.uint32)
            grp.attrs["template"] = action_labels[id_action]["template"]
            grp.create_dataset("videos", data=np.array(action_instances[id_action], dtype=np.uint32))
        
        # Prepare video labels.
        dset_videos = f.create_group("videos")
        for id_video in tqdm.tqdm(video_labels):
            label = video_labels[id_video]
            grp = dset_videos.create_group(str(id_video))
            grp.attrs.create("id_video", id_video, dtype=np.uint32)
            grp.attrs.create("id_action", label["id_action"], dtype=np.uint32)
            
            O = len(label["objects"])
            grp.attrs.create("objects", label["objects"], shape=(O,), dtype=h5py.string_dtype(encoding="utf-8"))
            
            # Get keyframes from actual video.
            keyframes_video = video_utils.get_keyframes(paths.data / f"videos/{id_video}.webm")
            keyframes = []
            boxes = []
            for keyframe in label["frames"]:
                if not keyframe in keyframes_video:
                    continue
                keyframes.append(keyframe)
                boxes_t = np.full((1 + O, 4), -float("inf"), dtype=np.float32)
                for obj, bbox in label["frames"][keyframe].items():
                    idx_obj = twentybn.utils.object_id_to_idx(obj)
                    boxes_t[idx_obj] = np.array(bbox, dtype=np.float32).flatten()
                boxes.append(boxes_t)
            
            grp.create_dataset("keyframes", data=keyframes, dtype=np.uint32)
            grp.create_dataset("boxes", data=boxes, shape=(len(boxes), 1 + O, 4), dtype=np.float32)
        
        f.create_dataset("video_ids", data=list(video_labels.keys()), dtype=np.uint32)



def find_continuous_ones(x: np.ndarray, left_to_right: bool):
    """Finds the maximum run of consecutive ones in the array.
        
        Args:
            x: 1d array.
            left_to_right: Whether to break ties with elements from left to right
        Returns:
            Range of largest run of consecutive ones (idx_start, idx_end).
    """
    x = np.concatenate((np.array([0]), x, np.array([0])))
    diff = x[1:] - x[:-1]
    start = np.squeeze(np.argwhere(diff == 1), axis=1)
    end = np.squeeze(np.argwhere(diff == -1), axis=1)
    
    idx = np.arange(len(start))
    if left_to_right:
        idx = idx[::-1]
    
    if left_to_right: 
        unsorted = np.array([(len(start) - i, end[i] - start[i]) for i in range(len(start))], dtype=[("idx", np.uint32), ("val", np.float32)])
    else:
        unsorted = np.array([(i, end[i] - start[i]) for i in range(len(start))], dtype=[("idx", np.uint32), ("val", np.float32)])
    idx_ranges = np.argsort(unsorted, order=("val", "idx"))[::-1]
    
    ranges = np.stack((start[idx_ranges], end[idx_ranges]), axis=0)
    return ranges

def find_pre_post_boundaries(x_class: np.ndarray) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
    """Finds the last certain pre-condition index and first certain post-condition index in their respective clusters.
    
    Args:
        x: [T] float32 class predictions (0-1).
    Returns:
        (last pre index, first post index).
    """
    pre_clusters = find_continuous_ones(x_class < 0.5, left_to_right=True)
    post_clusters = find_continuous_ones(x_class > 0.5, left_to_right=False)
#     print("pre_clusters:", pre_clusters)
#     print("post_clusters:", post_clusters)
    
    # Relax the constraints if one of the clusters is empty.
    if pre_clusters.size == 0:
        pre_clusters = find_continuous_ones(x_class <= 0.5, left_to_right=True)
    if post_clusters.size == 0:
        post_clusters = find_continuous_ones(x_class >= 0.5, left_to_right=False)
    
    idx_pre = 0
    idx_post = 0
    if pre_clusters.size == 0 and post_clusters.size == 0:
        # No clusters.
        return (0, x_class.shape[0])
    elif pre_clusters.size == 0:
        # Only post cluster.
        return (0, post_clusters[0, idx_post])
    elif post_clusters.size == 0:
        # Only pre cluster.
        return (pre_clusters[1, idx_pre], x_class.shape[0])
    
    while pre_clusters[0, idx_pre] >= post_clusters[1, idx_post]:
        if idx_pre >= pre_clusters.shape[1] - 1 and idx_post >= post_clusters.shape[1] - 1:
            return None, None
        
        # Avoid going past the last cluster.
        if idx_pre >= pre_clusters.shape[1] - 1:
            idx_post += 1
            continue
        elif idx_post >= post_clusters.shape[1] - 1:
            idx_pre += 1
            continue
        
        # Keep the larger cluster.
        size_pre = pre_clusters[1, idx_pre] - pre_clusters[0, idx_pre]
        size_post = post_clusters[1, idx_post] - post_clusters[0, idx_post]
        if size_pre > size_post:
            idx_post += 1
            continue
        elif size_pre < size_post:
            idx_pre += 1
            continue

        # Clusters have the same size. Advance the one with the larger succeeding cluster.
        size_pre = pre_clusters[1, idx_pre + 1] - pre_clusters[0, idx_pre + 1]
        size_post = post_clusters[1, idx_post + 1] - post_clusters[0, idx_post + 1]
        if size_pre >= size_post:
            idx_pre += 1
            continue
        elif size_post > size_pre:
            idx_post += 1
            continue
    
    # Make sure pre comes before post.
    post_clusters[0, idx_post] = max(post_clusters[0, idx_post], pre_clusters[0, idx_pre])
    pre_clusters[1, idx_pre] = min(pre_clusters[1, idx_pre], post_clusters[1, idx_post])
    
    return (pre_clusters[1, idx_pre], post_clusters[0, idx_post])

def test_results_to_probabilities(test_results: np.ndarray) -> np.ndarray:
    """Converts [2, T] (pre/post) test results where {0=false, 1=true, -1=unknown}
    to a [2, T] probability vector where {0=pre, 1=post, and 0.5=unknown}.
    
    Args:
        test_results: [2, T] (pre/post, num_timesteps) int32 condition test results.
    Returns:
        [2, T] (pre/post, num_timesteps) float32 probability.
    """
    x = np.array(test_results, dtype=np.float32)
    x_pre = x[0]
    x_post = x[1]
    
    # Set uncertain timesteps leaning to one side.
    idx_maybe_pre_post = (x_pre < 0) & (x_post > 0)
    idx_pre_maybe_post = (x_pre > 0) & (x_post < 0)
    idx_not_pre_maybe_post = (x_pre == 0) & (x_post < 0)
    idx_maybe_pre_not_post = (x_pre < 0) & (x_post == 0)
    idx_not_pre_post = idx_maybe_pre_post | idx_not_pre_maybe_post
    idx_pre_not_post = idx_pre_maybe_post | idx_maybe_pre_not_post
    x[:, idx_not_pre_post] = np.array([0.25, 0.75])[:, None]
    x[:, idx_pre_not_post] = np.array([0.75, 0.25])[:, None]
    
    # Set timesteps where both pre- and post-conditions are true.
    x[:, (x == 1).all(axis=0)] = 0.5
    
    # Set timesteps where neither pre- nor post-conditions are known.
    x[x < 0] = 0.5
    
    return x

def find_pre_post_frames(test_results: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Computes pre- and post-frames from the test results.
    
    Args:
        test_results: [2, T] int32 condition test results.
    Returns:
        (pre-frames, post-frames).
    """
    # [2, T]
    x_prob = test_results_to_probabilities(test_results)
#     print("x_prob:", x_prob)
    
    # Only choose non-zero elements.
    # [T]
    idx_valid = (x_prob != 0).any(axis=0)
    if (idx_valid == 0).all():
        return np.zeros((0,), dtype=np.uint32), np.zeros((0,), dtype=np.uint32)
    
    # Find pre/post boundaries among non-zero elements.
    # [NZ]
    x_class = x_prob[1, idx_valid]
    idx_nonzero = np.array(idx_valid.nonzero()[0], dtype=np.uint32)
    idx_pre_post = find_pre_post_boundaries(x_class)
    if idx_pre_post[0] is None or idx_pre_post[1] is None:
        return np.zeros((0,), dtype=np.uint32), np.zeros((0,), dtype=np.uint32)
#     print("idx_nonzero:", idx_nonzero)
#     print("idx_pre_post:", idx_pre_post)
    
    # Set boundary as mean between last pre and first post index in NZ.
    idx_boundary = int(0.5 * (idx_pre_post[0] + idx_pre_post[1]) + 0.5)
#     print("idx_boundary:", idx_boundary)
    
    # Convert NZ index to timestep.
    assert idx_boundary <= len(idx_nonzero)
    idx_boundary = idx_nonzero[min(len(idx_nonzero) - 1, idx_boundary)]
    
    # Set pre/post frames to uncertain frames (0.5) within the pre/post boundary.
    # (num_pre, num_post)
    idx_pre = idx_nonzero[(x_class <= 0.5) & (idx_nonzero < idx_boundary)]
    idx_post = idx_nonzero[(x_class >= 0.5) & (idx_nonzero >= idx_boundary)]
    
    return idx_pre, idx_post

def append_pre_post_to_dataset(results: typing.Dict[int, np.ndarray], paths: config.EnvironmentPaths, id_action: typing.Optional[int] = None):
    """Appends pre/post frames to the hdf5 dataset.
    
    Args:
        results: Test results in `condition_test_results.pkl`.
        paths: Environment paths.
        id_action: Process only this action, if not None.
    """
    with h5py.File(paths.data / "labels.hdf5", "a") as f:
        if id_action is None:
            id_videos = np.array(f["video_ids"])
        else:
            id_videos = np.array(f[f"actions/{id_action}/videos"])

        for id_video in tqdm.notebook.tqdm(id_videos):
            idx_pre, idx_post = find_pre_post_frames(results[id_video])

            grp_video = f["videos"][str(id_video)]
            if "pre" in grp_video:
                del grp_video["pre"]
            if "post" in grp_video:
                del grp_video["post"]
            grp_video.create_dataset("pre", data=idx_pre, dtype=np.uint32)
            grp_video.create_dataset("post", data=idx_post, dtype=np.uint32)



# generate splits



def generate_dataset_splits(
    pddl: symbolic.Pddl,
    stats: pd.DataFrame,
    twentybn_train_set: typing.List[int],
    twentybn_val_set: typing.List[int],
) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
    """Generates train, val, and test sets.
    
    Train and val sets are taken from the original 20BN train set, randomly selected for each action.
    Test set is taken from the original 20BN val set. The final splits are roughly (85, 15, 15).
    
    Args:
        pddl: Pddl instance.
        stats: Table output by `compute_condition_statistics()`.
    Returns:
        (train_set, val_set, test_set) 3-tuple.
    """
    random.seed(0)
    
    TRAIN_VAL = 1 - 0.15 / 0.85  # Assume original train set is 0.85 of the total.
    
    train_set = []
    val_set = []
    test_set = []
    
    A = len(pddl.actions)
    df = {
        "Action": [],
        "Distribution": [],
        "Dataset": [],
    }

    for id_action, action in enumerate(pddl.actions):
        s_partial = dnf_utils.get_partial_state(pddl, str(action))
        if s_partial.sum() == 0:
            df["Action"] += [id_action, id_action, id_action]
            df["Distribution"] += [0, 0, 0]
            df["Dataset"] += ["train", "val", "test"]
            continue

        stats_a = stats[stats.Action == id_action]
        stats_a = stats_a[(stats_a.Pre > 0) & (stats_a.Post > 0)]

        train_val_ids = list(stats_a.Video[stats_a.Dataset == "train"])
        test_ids = list(stats_a.Video[stats_a.Dataset == "val"])
        
        random.shuffle(train_val_ids)
        train_val_split = int(TRAIN_VAL * len(train_val_ids) + 0.5)
        train_ids = train_val_ids[:train_val_split]
        val_ids = train_val_ids[train_val_split:]

        train_set += train_ids
        val_set += val_ids
        test_set += test_ids
        
        num_train, num_val, num_test = len(train_ids), len(val_ids), len(test_ids)
        num_total = num_train + num_val + num_test
        df["Action"] += [id_action, id_action, id_action]
        df["Distribution"] += [num_train / num_total, num_val / num_total, num_test / num_total]
        df["Dataset"] += ["train", "val", "test"]
    
    plt.subplots(figsize=(5, 40))
    sns.barplot(data=df, y="Action", x="Distribution", hue="Dataset", orient="h")
    plt.xlabel("Proportion of dataset")
    plt.ylabel("Action")
    plt.title("Dataset distribution")
    
    # Preserve original dataset order.
    train_set, val_set, test_set = set(train_set), set(val_set), set(test_set)
    train_set = [id_video for id_video in twentybn_train_set if id_video in train_set]
    val_set = [id_video for id_video in twentybn_train_set if id_video in val_set]
    test_set = [id_video for id_video in twentybn_val_set if id_video in test_set]


    # pddl = symbolic.Pddl(str(paths.domain_pddl), str(paths.problem_pddl))
    # train_set, val_set, test_set = generate_dataset_splits(pddl, stats, twentybn_train_set, twentybn_val_set)
    
    # print(f"Train: {len(train_set)}")
    # print(f"Val: {len(val_set)}")
    # print(f"Test: {len(test_set)}")
    
    # with open(paths.data / "dataset_splits.pkl", "wb") as f:
    #     pickle.dump((train_set, val_set, test_set), f)
    
    return train_set, val_set, test_set




# predicate dataset



def create_predicate_dataset(
    pddl: symbolic.Pddl,
    labels: twentybn.dataset.Labels,
    dataset: typing.List[int],
    filename: str,
    path: pathlib.Path
):
    """Extracts pre/post frames and save them to an hdf5 dataset.
    
    Args:
        pddl: Pddl instance.
        labels: 20BN labels.
        dataset: List of video ids from the train/val set.
        filename: Name of dataset. The file will be saved as filename.hdf5.
        path: Path of dataset.
    """
    def extract_pre_post_worker(id_video: int):
        video_label = labels.videos[id_video]
        if video_label.pre.size == 0 or video_label.post.size == 0:
            return

        # Get pre/post video frames.
        path_video = path / f"videos/{id_video}.webm"
        keyframes = video_label.keyframes[np.concatenate((video_label.pre, video_label.post))]
        images = video_utils.read_video(path_video, keyframes)
        
        t_post = len(video_label.pre)
        pre_images = images[:t_post]
        post_images = images[t_post:]
        
        # [T, 16, 3, 4] (num_selected_frames, num_arg_combos, roi/arg_a/arg_b, x1/y1/x2/y2)
        pre_boxes = twentybn.utils.split_bbox_args(pddl, video_label, video_label.pre)
        post_boxes = twentybn.utils.split_bbox_args(pddl, video_label, video_label.post)
        
        grp = f.create_group(str(id_video))
        dset_pre_frames = grp.create_dataset("pre_frames", data=video_label.pre, dtype=np.uint32)
        dset_post_frames = grp.create_dataset("post_frames", data=video_label.post, dtype=np.uint32)
        
        H, W = pre_images[0].shape[:2]
        T_pre = len(pre_images)
        T_post = len(post_images)
#         print(len(pre_images), pre_images[0].shape, (T_pre, H, W, 3))
        dset_pre_images = grp.create_dataset("pre_images", data=pre_images, shape=(T_pre, H, W, 3), chunks=(1, H, W, 3), dtype=np.uint8, **hdf5plugin.Blosc(cname="lz4hc"))
        dset_post_images = grp.create_dataset("post_images", data=post_images, shape=(T_post, H, W, 3), chunks=(1, H, W, 3), dtype=np.uint8, **hdf5plugin.Blosc(cname="lz4hc"))
        
        dset_pre_boxes = grp.create_dataset("pre_boxes", data=pre_boxes, dtype=np.float32)
        dset_post_boxes = grp.create_dataset("post_boxes", data=post_boxes, dtype=np.float32)
    
    def collect_written_videos(f: h5py.File, dataset: typing.List[int]):
        """Record output videos and actions to the dataset.

        Args:
            f: Dataset file.
            dataset: List of video ids from the train/val set.
        """
        videos = []
        actions = []
        for id_video in dataset:
            if not str(id_video) in f.keys():
                continue
            videos.append(id_video)
            actions.append(labels.videos[id_video].id_action)

        dset_videos = f.create_dataset("videos", data=videos, dtype=np.uint32)
        dset_actions = f.create_dataset("actions", data=actions, dtype=np.uint32)
    
    with h5py.File(path / f"{filename}.hdf5", "w") as f:
        for id_video in tqdm.notebook.tqdm(dataset):
            extract_pre_post_worker(id_video)
        
        collect_written_videos(f, dataset)







# written videos





random.seed(0)

"""
dataset.hdf5 = {
  "id_video": [
    "images": [2, 3 + num_objects, H, W], uint8
    "boxes": [num_objects, 4] (x1, y1, x2, y2), float32
  ],
  "videos": [int, ...],
  "actions": [int, ...]
}
"""

def collect_written_videos(f: h5py.File, dataset: typing.List[int]):
    """Record output videos and actions to the dataset.
    
    Args:
        f: Dataset file.
        dataset: List of video ids from the train/val set.
    """
    videos = []
    actions = []
    for id_video in dataset:
        if not str(id_video) in f.keys():
            continue
        videos.append(id_video)
        actions.append(video_labels[id_video]["id_action"])

    dset_videos = f.create_dataset("videos", (len(videos),), dtype=np.uint32)
    dset_videos[:] = videos
    dset_actions = f.create_dataset("actions", (len(videos),), dtype=np.uint8)
    dset_actions[:] = actions

def extract_pre_post(dataset: typing.List[int], filename: str, path: pathlib.Path) -> typing.Dict[int, typing.Tuple]:
    """Extract pre/post frames and save them to an hdf5 dataset.
    
    Args:
        dataset: List of video ids from the train/val set.
        filename: Name of dataset. The file will be saved as filename.hdf5.
        path: Path of dataset.
    """
    def extract_pre_post_worker(id_video: int):
        if id_video not in video_ranges:
            return

        # Get pre/post video frames.
        path_video = path / f"videos/{id_video}.webm"
        keyframes = list(video_labels[id_video]["frames"].keys())
        pre_frames, post_frames = video_ranges[id_video]
        selected_keyframes = [random.choice(pre_frames), random.choice(post_frames)]
        
        try:
            pre_post_frames = video_utils.read_video(path_video, selected_keyframes)
        except:
            return
        if pre_post_frames is None:
            return

        # Write pre/post frames to dataset.
        height, width = pre_post_frames.shape[1:3]
        masks, indexed_boxes = twentybn.utils.create_bbox_masks(id_video, (height, width), video_labels, selected_keyframes)
        boxes = indexed_boxes[:, :, 1:]
        
        # ([2, 3, H, W], [2, 4, H, W]) => [2, 7, H, W]
        images = np.concatenate((np.moveaxis(pre_post_frames, 3, 1), masks), axis=1)
        
        grp = f.create_group(str(id_video))
        dset_images = grp.create_dataset("images", images.shape, dtype=np.uint8)
        dset_images[...] = images
        dset_boxes = grp.create_dataset("boxes", boxes.shape, dtype=np.float32)
        dset_boxes[...] = boxes
    
    with h5py.File(path / f"{filename}.hdf5", "w") as f:
        for id_video in tqdm.tqdm(dataset):
            extract_pre_post_worker(id_video)
#         with concurrent.futures.ThreadPoolExecutor(60) as pool:
#             futures = [pool.submit(extract_pre_post_worker, id_video) for id_video in dataset]
#             with tqdm.tqdm(total=len(dataset)) as pbar:
#                 for result in concurrent.futures.as_completed(futures):
#                     pbar.update(1)
        
        collect_written_videos(f, dataset)







# build











def build_dataset():
    
    paths = config.EnvironmentPaths(environment="twentybn")

    sth_else = load_something_else(paths)

    print(paths.data)
    
    """
    sth_sth_labels = {
        "{id_action}": "Holding something next to something"
    }
    """
    with open(paths.data / "labels/labels.json", "r") as f:
        sth_sth_labels = json.load(f)
    
    """
    sth_sth = [
        {
            "id": "78687",
            "label": "holding potato next to vicks vaporub bottle",
            "template": "Holding [something] next to [something]",
            "placeholders": ["potato", "vicks vaporub bottle"],
        }
    ]
    """
    with open(paths.data / "labels/train.json", "r") as f:
        sth_sth_train = json.load(f)
    
    with open(paths.data / "labels/validation.json", "r") as f:
        sth_sth_val = json.load(f)

    
    # Create template => idx_action map.
    idx_actions = {}
    for sth_sth_label in sth_sth_val:
        fine_label = sth_sth_label["label"]
        template = sth_sth_label["template"]
        
        coarse_label = re.sub("[\[\]]", "", template)
        idx_action = int(sth_sth_labels[coarse_label])
        idx_actions[template] = idx_action
    
    # Create action labels.
    """
    action_labels = [
        {
            "label": "Approaching something with your camera",
            "template": "Approaching [something] with your camera",
        }
    ]
    """
    action_labels = [None] * len(sth_sth_labels)
    for sth_sth_label in sth_sth_train:
        if not None in action_labels:
            break
        
        template = sth_sth_label["template"]
        coarse_label = re.sub("[\[\]]", "", template)
        idx_action = idx_actions[template]
        
        action_labels[idx_action] = {
            "label": coarse_label,
            "template": template
        }
    
    
    """
    video_labels = {
        {id_video}: {
            "id_action": id_action,
            "placeholders": ["a potato", "a vicks vaporub bottle"],
            "objects": ["potato", "bottle"],
            "frames": {
                idx_frame: {
                    "{id_object/hand}": [[x1, y1], [x2, y2]],
                },
            },
        },
    }
    train_set = [{video_id}, ...]
    val_set = [{video_id}, ...]
    """
    unsorted_video_labels = {}
    train_set = process_labels(paths.data / "train_set.pkl", sth_sth_train, sth_else, idx_actions, unsorted_video_labels)
    val_set = process_labels(paths.data / "val_set.pkl", sth_sth_val, sth_else, idx_actions, unsorted_video_labels)
    
    video_labels = {}
    for key in sorted(unsorted_video_labels.keys()):
        video_labels[key] = unsorted_video_labels[key]
    
    # Create action instances map.
    """
    action_instances = [
        [{id_video}, ...]
    ]
    """
    action_instances = [[] for _ in range(len(action_labels))]
    for id_video, video_label in video_labels.items():
        id_action = video_label["id_action"]
        action_instances[id_action].append(id_video)


    create_video_labels_dataset(paths, video_labels, action_labels, action_instances)
    return train_set, val_set, video_labels, action_instances


# Create video labels.
def process_labels(path, sth_sth_set, sth_else, idx_actions, video_labels):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    labels = []
    for sth_sth_label in tqdm.tqdm(sth_sth_set):
        id_video = int(sth_sth_label["id"])
        if not str(id_video) in sth_else:
            continue

        template = sth_sth_label["template"]
        id_action = idx_actions[template]
        placeholders = sth_sth_label["placeholders"]

        sth_else_label = sth_else[str(id_video)]
        objects = sth_else_label[0]["gt_placeholders"]

        frames = {}
        for sth_else_frame in sth_else_label:
            idx_frame = int(re.match(r"\d+/(\d+)\.jpg", sth_else_frame["name"])[1]) - 1
            boxes = {}
            for sth_else_box in sth_else_frame["labels"]:
                idx_obj = sth_else_box["standard_category"]
                if idx_obj != "hand":
                    # Simplify integer. JSON key value still need to be strings.
                    idx_obj = str(int(idx_obj))

                box = sth_else_box["box2d"]
                boxes[idx_obj] = [[box["x1"], box["y1"]], [box["x2"], box["y2"]]]
            
            frames[idx_frame] = boxes

        labels.append(id_video)
        video_labels[id_video] = {
            "id_action": id_action,
            "placeholders": placeholders,
            "objects": objects,
            "frames": frames,
        }
    with open(path, "wb") as f:
        pickle.dump(labels, f)
    return labels

