import os
import cv2
from .eye import Eye  # Keep for future
from .calibration import Calibration


class GazeTracking:
    """
    Simplified gaze tracking class using OpenCV's Haar cascade for face detection.
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # Using OpenCV's Haar cascade for face detection
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    @property
    def pupils_located(self):
        return False  # Can't track pupils without landmarks

    def _analyze(self):
        """Detects the face using OpenCV (no eye or pupil tracking)."""
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            self.eye_left = None
            self.eye_right = None
        else:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        return None

    def pupil_right_coords(self):
        return None

    def horizontal_ratio(self):
        return None

    def vertical_ratio(self):
        return None

    def is_right(self):
        return False

    def is_left(self):
        return False

    def is_center(self):
        return False

    def is_blinking(self):
        return False

    def annotated_frame(self):
        frame = self.frame.copy()
        return frame
