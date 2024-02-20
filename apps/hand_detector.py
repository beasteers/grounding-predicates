import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.python.solutions import hands as mp_hands

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), 'hand_landmarker.task'))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass    

    def detect(self, img):
        # print("detecting image", type(img), img.shape)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        r = self.detector.detect(mp_image)
        return r

class Hand:
    def __init__(self, hand_landmarks):
        # Store the hand landmarks
        # self.hand_landmarks = hand_landmarks
        # from IPython import embed
        # embed()
        # if not isinstance(hand_landmarks, landmark_pb2.NormalizedLandmarkList):
        #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #     hand_landmarks_proto.landmark.extend()
        # else:
        #     hand_landmarks_proto = hand_landmarks[0]
        # print(hand_landmarks)
        self.hand_landmarks = [
            # landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z)
            if hasattr(l, 'x') else
            landmark_pb2.NormalizedLandmark(x=l[0], y=l[1], z=l[2])
            for l in hand_landmarks
        ]
        self.image_height = 240
        self.image_width = 427

    def fingertips(self):
        # Extract and return the fingertip landmarks
        fingertips = []
        for i in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]:
            fingertips.append(np.array([self.hand_landmarks[i].x*self.image_width,self.hand_landmarks[i].y*self.image_height]))
        return np.array(fingertips)

    def fingers(self):
        # Extract and return the fingertip landmarks
        fingers = []
        thumb = [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP]
        index = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = [mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring = [mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = [mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP]
        for finger in [thumb, index, middle, ring, pinky]:
          for i in finger:
            fingers.append(np.array([self.hand_landmarks[i].x*self.image_width,self.hand_landmarks[i].y*self.image_height]))
        return np.array(fingers)

    def palm(self):
        # Extract and return the palm landmarks
        palm = []
        for i in [mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]:
            palm.append(np.array([self.hand_landmarks[i].x*self.image_width,self.hand_landmarks[i].y*self.image_height]))
        return np.array(palm)


