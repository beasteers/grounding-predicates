import typing

# import sys
# sys.path.append("/scratch/data/repos/grounding-predicates")

import numpy as np
import symbolic

from apps import hand_detector
from env import twentybn
from gpred import video_utils, dnf_utils
import config
from predicate_tests import generate_tests


def draw_hands(img: np.ndarray, detected_hands: typing.List[hand_detector.Hand]):
    import PIL
    
    img = PIL.Image.fromarray(img)
    draw = PIL.ImageDraw.Draw(img)

    for hand in detected_hands:
        draw.polygon(hand.palm().flatten().tolist(), outline=(0,255,0))
        for xy in hand.palm():
            box = np.concatenate([xy - 10, xy + 10], axis=0)
            draw.ellipse(box.tolist(), outline=(255,0,0))
        for finger in hand.fingers():
            draw.line(finger.flatten().tolist(), fill=(255,0,255))
            xy = finger[-1]
            box = np.concatenate([xy - 15, xy + 15], axis=0)
            draw.ellipse(box.tolist(), outline=(255,255,255))
            
    img = np.array(img)
    
    return img

def evaluate_video_conditions(
    paths: config.EnvironmentPaths,
    pddl: symbolic.Pddl,
    video_label: twentybn.dataset.VideoLabel,
    hands: np.ndarray,
    id_video: int,
    s_partial: np.ndarray,
    tests: typing.List[typing.Tuple[int, typing.Callable[[np.ndarray, hand_detector.Hand, bool], bool]]],
    generate_video: bool = False,
) -> np.ndarray:
    """Evaluates the pre/post-conditions for the given video.
    
    Args:
        paths: Environment paths.
        pddl: Pddl instance.
        video_label: 20BN label.
        hands: [T] (num_keyframes) list of [H] (num_hands) lists of [L, 2] (num_landmarks, x/y) landmark arrays for given video.
        id_video: Video id.
        s_partial: [2, 2, N] (pre/post, pos/neg, num_props) Partial state for current action.
        tests: Output of `generate_tests()`.
        generate_video: Whether to generate a video with the object/hand detections.
    Returns:
        [2, T] (pre/post, num_frames) int array indicating whether the frame passes the condition tests (0: False, 1: True, -1: Unknown).
    """

    # Get test propositions.
    # [2, 2, N] -> [2, 2, Q] (pre/post, pos/neg, num_tests)
    idx_props = [idx_prop for idx_prop, test in tests]
    s_expected = s_partial[:, :, idx_props]
    prop_labels = ["pre", "post"] + [pddl.state_index.get_proposition(idx_prop) for idx_prop in idx_props]
    
    # [2, 2, Q] -> [2, Q] (pre/post, num_tests)
    s_expected_pos = s_expected[:, 0, :]
    s_expected_neg = s_expected[:, 1, :]
    
    # Prepare output.
    # [2, T] (pre/post, num_frames)
    T = len(hands)
    test_results = np.zeros((2, T), dtype=np.int8)
    
    if generate_video:
        # Load video.
        video_frames = video_utils.read_video(paths.data / "videos" / f"{id_video}.webm", video_label.keyframes)
        
        video_out = []
    
    # Iterate over all keyframes.
    for t in range(T):
        box_hand = video_label.boxes[t, 0, :]
        if box_hand[0] >= 0:
#                 xy1_hand = np.maximum(0, box_hand[:2].astype(np.int) - 100)
#                 xy2_hand = np.minimum(img.shape[:2][::-1], (box_hand[2:] + 101).astype(np.int))
#                 img_hand = img[xy1_hand[1]:xy2_hand[1], xy1_hand[0]:xy2_hand[0]]
#                 detected_hands = hands.detect(img_hand, xy_offset=xy1_hand)
            # detected_hands = load_detected_hands(hands[t])
            detected_hands = [hand_detector.Hand(hand_landmark) for hand_landmark in hands[t]]
        else:
            detected_hands = []

        # hand = detected_hands[0] if detected_hands else None
        hand = identify_contained_hand(box_hand, detected_hands)

        # Run tests.
        # [2, Q] (pos/neg, num_tests)
        s_results = precompute_tests(tests, video_label.boxes[t], hand)

        # [2, Q] -> [Q]
        s_results_pos = s_results[0, :]
        s_results_neg = s_results[1, :]

        # [2, Q] (pre/post, num_tests)
        violated = (s_expected_pos & s_results_neg[None, :]) | (s_expected_neg & s_results_pos[None, :])
        unknown = (s_expected_pos & ~s_results_pos[None, :]) | (s_expected_neg & ~s_results_neg[None, :])

        # [2] (pre/post)
        satisfied = np.ones((violated.shape[0],), dtype=np.int8)
        satisfied[unknown.any(axis=1)] = -1
        satisfied[violated.any(axis=1)] = 0
        test_results[:, t] = satisfied

        if generate_video:
            # Load video frame.
            img = video_frames[t]
            
            # Draw hands/bounding boxes.
            img = draw_hands(img, detected_hands)
            img = video_utils.draw_bounding_boxes(img, video_label.boxes[t], ["hand"] + video_label.objects)

            # Convert condition test results to probabilities.
            # [2]
            p_conditions = test_results[:, t].astype(np.float32)
            p_conditions[p_conditions < 0] = 0.5

            # Convert test values to probabilities.
            # [Q]
            p_results = s_results_pos.astype(np.float32) - s_results_neg.astype(np.float32)
            p_results = 0.5 * (p_results + 1)

            # Show pre/post-condition timesteps if available.
            prop_labels_t = prop_labels.copy()
            if t in video_label.pre:
                prop_labels_t[0] = "pre   !!!!!!!!!!!!!!!!!!!!"
            elif t in video_label.post:
                prop_labels_t[1] = "post !!!!!!!!!!!!!!!!!!!!"

            # [2], [Q] -> [2 + Q]
            p_predict = np.concatenate((p_conditions, p_results), axis=0)
            img = video_utils.overlay_predictions(img, p_predict, prop_labels_t)
            video_out.append(img)
    
    if generate_video:
        video_utils.write_video(paths.data / "labeled_videos" / f"{id_video}.webm", video_out)
    
    return test_results


def process_action(id_action: int, generate_video: bool = False, num_videos: typing.Optional[int] = None) -> typing.Dict[int, np.ndarray]:
    """Checks the pre/post-conditions for all the videos for one action.
    
    Assumes `initialize_tests()` has already been called.
    
    Args:
        id_action: Action id.
        generate_video: Whether to generate a video for visualization.
        num_videos: Maximum number of videos per action to process.
    Returns:
        Map from id_video to [2, T] (pre/post, num_frames) int array indicating whether the frame passes the condition tests (0: False, 1: True, -1: Unknown).
    """
    global paths, labels, pddl, tests, hands

    action = str(pddl.actions[id_action])
    s_partial = dnf_utils.get_partial_state(pddl, action)
    
    # Iterate over all videos of the action.
    test_results = {}
    id_videos = labels.actions[id_action].videos if num_videos is None else labels.actions[id_action].videos[:num_videos]
    for id_video in id_videos:
        try:
            test_results[id_video] = evaluate_video_conditions(paths, pddl, labels.videos[id_video], hands[id_video], id_video, s_partial, tests, generate_video)
        except Exception as e:
            print(f"id_action={id_action}:\n{e}")
            with open(f"{id_action}.log", "a") as f:
                f.write(f"{id_video}:\n{e}\n")
    
    return test_results


def evaluate_pre_post(pddl, paths):
    # with open(paths.data / "condition_test_results.pkl", "rb") as f:
    #     test_results = pickle.load(f)

    tests = generate_tests(pddl)
    test_results = {}
    for id_action in tqdm.tqdm(range(len(pddl.actions))):
        test_results.update(process_action(id_action))
    with open(paths.data / "condition_test_results.pkl", "wb") as f:
        pickle.dump(test_results, f)
    return tests, test_results