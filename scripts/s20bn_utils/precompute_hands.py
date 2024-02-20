from math import e
import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import tqdm
import typing
import pickle
import concurrent.futures
import numpy as np

import config
import symbolic
from env import twentybn
from gpred import video_utils
from apps import hand_detector

paths = config.EnvironmentPaths(environment="twentybn")
pddl = symbolic.Pddl(str(paths.env / "domain.pddl"), str(paths.env / "problem.pddl"))
labels = twentybn.dataset.Labels(paths.data / "labels.hdf5")



def precompute_hands(id_action: int) -> typing.Dict[int, typing.List[typing.List[np.ndarray]]]:
    """Precompues hand detections for all the videos for one action.

    Args:
        id_action: Action id.
    Returns:
        Map from id_video to [T] (num_keyframes) list of [H] (num_hands) list of [L, 2] (num_landmarks, x/y) float32 array of hand detections.
    """
    action = str(pddl.actions[id_action])
    
    # Iterate over all videos of the action.
    detected_hands = {}
    id_videos = labels.actions[id_action].videos
    print(f"Starting {id_action} {action} {len(id_videos)}\n", end="")
    try:
        with hand_detector.HandDetector(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            for i, id_video in enumerate(id_videos):
                video_hands = detected_hands[id_video] = []
                video_label = labels.videos[id_video]
                video_frames = video_utils.read_video(paths.data / "videos" / f"{id_video}.webm", video_label.keyframes)
                for t, img in enumerate(video_frames):
                    video_hands.append([hand.hand_landmarks for hand in hands.detect(img)])
                print(f'{id_action} {i}/{len(id_videos)} {id_video}\n', end='', flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    return detected_hands


import ipdb
@ipdb.iex
def load_hands(n_workers=6):
    # cached hand detections.
    hands = load_pickle(paths.data / "hands.pkl")
    if hands:
        return hands
    hands = load_pickle(paths.data / "hands_tmp.pkl") or {}

    try:
        if n_workers < 2:
            for id_action in tqdm.notebook.tqdm(range(len(pddl.actions))):
                hands.update(precompute_hands(id_action))
        else:
            # Process hand detections for all actions in parallel.
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for id_action in range(len(pddl.actions)):
                    print(id_action, pddl.state_index.get_proposition(id_action))
                    future = executor.submit(precompute_hands, id_action)
                    futures[future] = id_action
                
                with tqdm.tqdm(total=len(futures)) as loop:
                    for future in concurrent.futures.as_completed(futures):
                        id_action = futures[future]
                        try:
                            hands.update(future.result())
                            print("Finished", id_action, pddl.state_index.get_proposition(id_action))
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            print(f"Exception for id_action={id_action}:\n{e}", flush=True)
                            raise
                        loop.update(1)
    except:
        save_pickle(paths.data / "hands_tmp.pkl", hands)
    else:
        save_pickle(paths.data / "hands.pkl", hands)
    return hands


def load_pickle(path):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)

def save_pickle(path, hands):
    # Save hand detections.
    if len(hands):
        print("Saving", path, len(hands))
        with open(path, "wb") as f:
            pickle.dump(hands, f)

if __name__ == '__main__':
    import os, signal, atexit
    os.setpgrp() # create new process group, become its leader
    def cleanup():
        os.killpg(0, signal.SIGKILL)
    atexit.register(cleanup)

    import fire
    fire.Fire(load_hands)
    # load_hands(24)