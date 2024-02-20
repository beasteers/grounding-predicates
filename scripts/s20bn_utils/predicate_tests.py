import typing

# import sys
# sys.path.append("/scratch/data/repos/grounding-predicates")

import tqdm
import numpy as np
import symbolic

from apps import hand_detector
from env import twentybn
from gpred import video_utils, dnf_utils
import config


paths = config.EnvironmentPaths(environment="twentybn")
pddl = symbolic.Pddl(str(paths.env / "domain.pddl"), str(paths.env / "problem.pddl"))




def point_inside_rectangle(box: np.ndarray, points: np.ndarray) -> typing.Union[bool, np.ndarray]:
    """Checks whether the points fall inside the rectangle.

    Args:
        box: [4] (x1/y1/x2/y2) corners.
        points: [N, 2] or [2] (x/y).
    Returns:
        Boolean if one point is given, array of booleans [N] otherwise.
    """
    # One point.
    if points.shape == (2,):
        return box[0] <= points[0] and points[0] <= box[2] and box[1] <= points[1] and points[1] <= box[3]
    return (box[0] <= points[:,0]) & (points[:,0] <= box[2]) & (box[1] <= points[:,1]) & (points[:,1] <= box[3])

def line_segment_circle_collision(line_segment: np.ndarray, circle: typing.Tuple[np.ndarray, float]) -> bool:
    """Checks whether the line segment and circle collide.
    
    Args:
        line_segment: [4] (x1/y1/x2/y2) endpoints.
        circle: ([2] (x/y) center, radius).
    Returns:
        True if the shapes collide.
    """
    # [[x1, y1], [x2, y2]]
    endpoints = np.reshape(line_segment, (2, 2))
    #print("endpoints:", endpoints)
    
    # [cx, cy]
    center, radius = circle
    #print("center:", center)
    r2 = radius * radius
    #print("radius2:", r2)
    
    # [[x1 - cx, y1 - cy], [x2 - cx, y2 - cy]]
    dc = center[None, :] - endpoints
    dd = np.sum(dc * dc, axis=1)
    idx_min = np.argmin(dd)
    #print("min:", idx_min, dd)
    
    # Check if closer endpoint is within radius.
    if dd[idx_min] < r2:
        return True
    
    # [x, y]
    origin = endpoints[idx_min]
    v_line = endpoints[1 - idx_min] - origin
    v_line /= np.linalg.norm(v_line)
    v_circle = center - origin
    
    # Check if projection of circle onto line falls outside the segment.
    d_circle_line = v_line.dot(v_circle)
    if d_circle_line < 0:
        #print(":", d_circle_line)
        return False
    
    # Orthogonal distance between circle and line.
    d_circle = v_circle - d_circle_line * v_line
    #print("::", d_circle.dot(d_circle), r2)
    return d_circle.dot(d_circle) < r2


def box_circle_collision(box: np.ndarray, circle: typing.Tuple[np.ndarray, float]) -> bool:
    """Checks whether the box and circle collide.
    
    Args:
        box: [4] (x1/y1/x2/y2) corners.
        circle: ([2] (x/y) center, radius).
    Returns:
        True if the shapes collide.
    """
    x1, y1, x2, y2 = box
    return (
        point_inside_rectangle(box, circle[0]) or
        line_segment_circle_collision(np.array([x1, y1, x1, y2]), circle) or
        line_segment_circle_collision(np.array([x2, y1, x2, y2]), circle) or
        line_segment_circle_collision(np.array([x1, y1, x2, y1]), circle) or
        line_segment_circle_collision(np.array([x1, y2, x2, y2]), circle)
    )

def box_box_collision(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    """Checks whether the two boxes collide.
    
    Args:
        box_a: [4] (x1/y1/x2/y2) corners.
        box_b: [4] (x1/y1/x2/y2) corners.
    Return:
        Whether the two boxes collide.
    """
    minkowski_0 = box_a[:2] - box_b[2:]
    minkowski_1 = box_a[2:] - box_b[:2]
    return (np.sign(minkowski_0) != np.sign(minkowski_1)).all()

def box_hand_collision(box: np.ndarray, hand: hand_detector.Hand, radius: float = 15) -> bool:
    """Checks whether the box overlaps with any of the fingertips.
    
    Args:
        box: [4] (x1/y1/x2/y2) corners.
        hand: Detected hand.
        radius: Distance from fingertips.
    Returns:
        Whether the box overlaps with any of the fingertips.
    """
    for fingertip in hand.fingertips():
        if box_circle_collision(box, (fingertip, radius)):
            return True
    return False

def identify_contained_hand(box: np.ndarray, detected_hands: typing.List[hand_detector.Hand]) -> typing.Optional[hand_detector.Hand]:
    """Identifies which hand is contained inside the bounding box.
    
    Args:
        box: [4] (x1/y1/x2/y2) corners.
        detected_hands: Detected hands output by `hand_detector.HandDetector`.
    Returns:
        Hand corresponding to the one in the bounding box if any.
    """
    
    is_contained = np.zeros((len(detected_hands),), dtype=int)
    for i, hand in enumerate(detected_hands):
        # print(len(hand.hand_landmarks))
        points = np.concatenate((hand.palm(), hand.fingertips()), axis=0)
        # print(points.shape)
        # input()
        is_contained[i] = point_inside_rectangle(box, points).sum()
    
    if is_contained.sum() == 0:
        return None
    
    idx_max = is_contained.argmax()
    return detected_hands[idx_max]

class PropositionTestFailure(Exception):
    def __init__(self, message):
        self.message = message

class PropositionUnknown(Exception):
    def __init__(self, message):
        self.message = message

def is_sth_visible(boxes: np.ndarray, idx_object: int, expected: typing.Optional[bool] = None) -> bool:
    """Checks whether the specified object is visible.
    
    Raises a PropositionTestFailure if the expected result is specified and does not match the test result.
    
    Args:
        boxes: [4, 4] (hand/a/b/c, x1/y1/x2/y2) box corners.
        idx_object: Object index (0/1/2/3 for "hand"/"a"/"b"/"c").
        expected: Expected result.
    """
    result = idx_object < boxes.shape[0] and boxes[idx_object, 0] >= 0
    
    if expected is not None and result != expected:
        raise PropositionTestFailure(f"visible({idx_object}) != {expected}")
    
    return result

def is_sth_touching_hand(
    boxes: np.ndarray,
    idx_object: int,
    hand: typing.Optional[hand_detector.Hand],
    expected: typing.Optional[bool] = None
) -> bool:
    """Checks whether the specified object is touching the hand.
    
    Raises a PropositionTestFailure if the expected result is specified and does not match the test result.
    
    Args:
        boxes: [4, 4] (hand/a/b/c, x1/y1/x2/y2) box corners.
        idx_object: Object index (0/1/2/3 for "hand"/"a"/"b"/"c").
        hand: Detected hand, if it exists.
        expected: Expected result.
    """
    if idx_object >= boxes.shape[0] or boxes[0, 0] < 0 or boxes[idx_object, 0] < 0:
        result = False
    elif hand is None:
        raise PropositionUnknown(f"touching({'abc'[idx_object-1]}, hand): No hand detected.")
    else:
        box_hand = boxes[0, :]
        box_obj = boxes[idx_object, :]

        # If boxes don't collide, then they are not touching.
        result = box_box_collision(box_hand, box_obj) and box_hand_collision(box_obj, hand)
    
    if expected is not None and result != expected:
        raise PropositionTestFailure(f"touching({'abc'[idx_object-1]}, hand) != {expected}")
    
    return result

def is_sth_touching_sth(boxes: np.ndarray, idx_object_a: int, idx_object_b, expected: typing.Optional[bool] = None) -> bool:
    """Checks whether one object is touching another.
    
    Returns false if the objects are not overlapping, otherwise raises a PropositionUnknown.
    Raises a PropositionTestFailure if the expected result is specified and does not match the test result.
    
    Args:
        boxes: [4, 4] (hand/a/b/c, x1/y1/x2/y2) box corners.
        idx_object_a: Object index (0/1/2/3 for "hand"/"a"/"b"/"c").
        idx_object_b: Object index (0/1/2/3 for "hand"/"a"/"b"/"c").
        expected: Expected result.
    """
    if max(idx_object_a, idx_object_b) >= boxes.shape[0] or boxes[idx_object_a, 0] < 0 or boxes[idx_object_b, 0] < 0:
        raise PropositionUnknown(f"touching({'abc'[idx_object_a-1]}, {'abc'[idx_object_b-1]}): Missing object.")
    
    box_a = boxes[idx_object_a, :]
    box_b = boxes[idx_object_b, :]
    # If boxes don't collide, then they are not touching.
    result = box_box_collision(box_a, box_b)
    if result:
        raise PropositionUnknown(f"touching({'abc'[idx_object_a-1]}, {'abc'[idx_object_b-1]}): Unable to determine from overlapping boxes.")
    
    if expected is not None and result != expected:
        raise PropositionTestFailure(f"touching({'abc'[idx_object_a-1]}, {'abc'[idx_object_b-1]}) != {expected}")
    
    return result







def generate_tests(
    pddl: symbolic.Pddl,
    s_partial: typing.Optional[np.ndarray] = None
) -> typing.List[typing.Tuple[int, typing.Callable[[typing.Dict, bool], bool]]]:
    """Generate tests for detecting start/end frames.
    
    The first test evaluates whether 'a' is touching the hand by testing whether their bounding boxes overlap.
    The remaining tests evaluates whether all the objects are visible.
    Only the tests specifically required by the action pre/post-conditions should be run.
    
    Args:
        pddl: Pddl instance.
        s_partial: [A, 2, 2, N] (action, pre/post, pos/neg, state) Partial states for all actions. If provided, this function will test which actions are not covered by the tests.
    Returns:
        List of (idx_prop, lambda boxes: bool) tuples where bounding boxes should be passed into the lambda to evaluate the test condition.
    """
    props = dnf_utils.get_index_propositions(pddl.state_index)
    idx_props_visible = [props.index(f"visible({obj})") for obj in ["a", "b", "c", "hand"]]
    idx_props_touching = [props.index(f"touching({sth_a}, {sth_b})") for sth_a, sth_b in [("a", "hand"), ("b", "hand"), ("c", "hand"), ("a", "b"), ("a", "c"), ("b", "c")]]

    tests_visible = [
        (idx_props_visible[0], lambda boxes, hand, expected: is_sth_visible(boxes, 1, expected)),
        (idx_props_visible[1], lambda boxes, hand, expected: is_sth_visible(boxes, 2, expected)),
        (idx_props_visible[2], lambda boxes, hand, expected: is_sth_visible(boxes, 3, expected)),
        (idx_props_visible[3], lambda boxes, hand, expected: is_sth_visible(boxes, 0, expected)),
    ]
    tests_touching = [
        (idx_props_touching[0], lambda boxes, hand, expected: is_sth_touching_hand(boxes, 1, hand, expected)),
        (idx_props_touching[1], lambda boxes, hand, expected: is_sth_touching_hand(boxes, 2, None, expected)),
        (idx_props_touching[2], lambda boxes, hand, expected: is_sth_touching_hand(boxes, 3, None, expected)),
        (idx_props_touching[3], lambda boxes, hand, expected: is_sth_touching_sth(boxes, 1, 2, expected)),
        (idx_props_touching[4], lambda boxes, hand, expected: is_sth_touching_sth(boxes, 1, 3, expected)),
        (idx_props_touching[5], lambda boxes, hand, expected: is_sth_touching_sth(boxes, 2, 3, expected)),
    ]
    
    if s_partial is not None:
        print("Actions not covered by tests:")
        idx_props = idx_props_touching + idx_props_visible
        for id_action, action in enumerate(pddl.actions):
            # [2, N]
            s_pre = s_partial[id_action,0,...]
            s_post = s_partial[id_action,1,...]
            if not s_pre[:,idx_props].any():
                print(id_action, actions[id_action])
    
    return tests_visible + tests_touching

def evaluate_tests(
    tests: typing.List[typing.Tuple[int, typing.Callable[[np.ndarray, hand_detector.Hand, bool], bool]]],
    s_partial: np.ndarray,
    boxes: np.ndarray,
    hand: hand_detector.Hand,
) -> bool:
    """Evaluates whether the propositions given by the tests are satisfied in the partial states.
    
    Any proposition not specified in the partial state is assumed to pass its corresponding test.
    
    Args:
        test: List of (idx_prop, lambda(oxes, hand, expectedd) -> bool) pairs.
        s_partial: [2, N] (pos/neg, num_props) Partial state.
        boxes: [4, 4] (hand/a/b/c, x1/y1/x2/y2) Bounding boxes of objects for the given frame.
        hand: Detected hand.
    Returns:
        True if all the tests are satisfied, raises a PropositionTestFailure otherwise.
    """
    s_pos = s_partial[0]
    s_neg = s_partial[1]
    
    for idx_prop, test in tests:
        if not s_pos[idx_prop] and not s_neg[idx_prop]:
            # Proposition not specified in partial state, so don't test.
            continue
            
        # Either pos or neg is true.
        expected = s_pos[idx_prop]
        test(boxes, hand, expected)
    return True

def precompute_tests(
    tests: typing.List[typing.Tuple[int, typing.Callable[[np.ndarray, hand_detector.Hand, bool], bool]]],
    boxes: np.ndarray,
    hand: hand_detector.Hand,
) -> np.ndarray:
    """Evaluates whether the propositions given by the tests are true.
    
    Any proposition not specified in the partial state is assumed to pass its corresponding test.
    
    Args:
        test: List of (idx_prop, lambda(boxes, hand, expectedd) -> bool) pairs.
        s_partial: [2, N] (pos/neg, num_props) Partial state.
        boxes: [4, 4] (hand/a/b/c, x1/y1/x2/y2) Bounding boxes of objects for the given frame.
        hand: Detected hand.
    Returns:
        [2, Q] (pos/neg, num_tests) Partial state over whether each test returns true.
    """
    results = np.zeros((2, len(tests)), dtype=bool)
    
    # Iterate over all tests.
    for idx_test, (idx_prop, test) in enumerate(tests):
        try:
            # Run test.
            val = test(boxes, hand, None)
        except PropositionUnknown as e:
            # Leave partial state as 0.
            continue

        idx_pos_neg = 1 - val
        results[idx_pos_neg, idx_test] = True
    
    return results




def draw_hands(img: np.ndarray, detected_hands):
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


def process_action(id_action: int, hands, tests, labels, *, generate_video: bool = False, num_videos: typing.Optional[int] = None) -> typing.Dict[int, np.ndarray]:
    """Checks the pre/post-conditions for all the videos for one action.
    
    Assumes `initialize_tests()` has already been called.
    
    Args:
        id_action: Action id.
        generate_video: Whether to generate a video for visualization.
        num_videos: Maximum number of videos per action to process.
    Returns:
        Map from id_video to [2, T] (pre/post, num_frames) int array indicating whether the frame passes the condition tests (0: False, 1: True, -1: Unknown).
    """
    action = str(pddl.actions[id_action])
    s_partial = dnf_utils.get_partial_state(pddl, action)
    
    # Iterate over all videos of the action.
    test_results = {}
    id_videos = labels.actions[id_action].videos if num_videos is None else labels.actions[id_action].videos[:num_videos]
    for id_video in id_videos:
        # try:
            test_results[id_video] = evaluate_video_conditions(paths, pddl, labels.videos[id_video], hands[id_video], id_video, s_partial, tests, generate_video)
        # except Exception as e:
        #     print(f"id_action={id_action}:\n{e}")
        #     with open(f"{id_action}.log", "a") as f:
        #         f.write(f"{id_video}:\n{e}\n")
    
    return test_results


def evaluate_pre_post(pddl, paths, hands, tests, labels):
    # with open(paths.data / "condition_test_results.pkl", "rb") as f:
    #     test_results = pickle.load(f)
    test_results = {}
    for id_action in tqdm.tqdm(range(len(pddl.actions))):
        test_results.update(process_action(id_action, hands, tests, labels))
    with open(paths.data / "condition_test_results.pkl", "wb") as f:
        pickle.dump(test_results, f)
    return tests, test_results

