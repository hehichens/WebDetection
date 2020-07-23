from __future__ import division
import os
import cv2
import dlib
import math
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "/home/hichens/Datasets/dat/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def hisEqulColor(self):
        ycrcb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        self.frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            self.face = faces[0]
            landmarks = self._predictor(frame, self.face)
            self.landmarks = landmarks
            self.left_point = [landmarks.part(36).x, landmarks.part(36).y]
            self.right_point = [landmarks.part(45).x, landmarks.part(45).y]
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

            self.x_left = self.eye_left.origin[0] + self.eye_left.pupil.x
            self.y_left = self.eye_left.origin[1] + self.eye_left.pupil.y
            self.x_right = self.eye_right.origin[0] + self.eye_right.pupil.x
            self.y_right = self.eye_right.origin[1] + self.eye_right.pupil.y

            self.X = [self.x_left - self.left_point[0] + self.y_left - self.right_point[0],
                      self.x_right - self.right_point[0] + self.y_right - self.right_point[0]]
            self.Y = [landmarks.part(30).x - landmarks.part(27).x, landmarks.part(30).y - landmarks.part(27).y]
            self.cosXY = (self.X[0] * self.Y[0] + self.X[1] * self.Y[1]) \
                         / (math.sqrt(self.X[0] ** 2 + self.X[1] ** 2) * math.sqrt(self.Y[0] ** 2 + self.Y[1] ** 2))


        except :
            self.eye_left = None
            self.eye_right = None


    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self.origin_frame = frame.copy()
        self.hisEqulColor()
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            return (self.x_left, self.y_left)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            return (self.x_right, self.y_right)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.origin_frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            face = self.face
            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            # draw the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

            x_left, y_left = self.left_point
            x_right, y_right = self.right_point
            cv2.circle(frame, (x_left, y_left), 2, color)
            cv2.circle(frame, (x_right, y_right), 2, color)

        return frame