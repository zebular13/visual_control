from dataclasses import dataclass
import sys
import time

import cv2
import mediapipe as mp

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CV_DRAW_COLOR_PRIMARY = (255, 255, 0)

CONTROL_CIRCLE_DEADZONE_R = 50

CONTROL_CIRCLE_XY_CENTER = (int(CAMERA_WIDTH/4), int(CAMERA_HEIGHT/2))
CONTROL_CIRCLE_Z_APERATURE_CENTER = (
    int(3*CAMERA_WIDTH/4), int(CAMERA_HEIGHT/2))


@dataclass
class HandData:
    handedness: str
    landmarks: list
    center_perc: tuple

    def __init__(self, handedness, landmarks):
        self.handedness = handedness
        self.landmarks = landmarks
        x_avg = sum(lm.x for lm in landmarks) / len(landmarks)
        y_avg = sum(lm.y for lm in landmarks) / len(landmarks)
        z_avg = sum(lm.z for lm in landmarks) / len(landmarks)
        self.center_perc = (x_avg, y_avg, z_avg)
        print(f"center_perc: {self.center_perc}")


def lerp(a, b, t):
    return a(b-a)*t


def draw_control_overlay(img, lh_data=None, rh_data=None):
    cv2.circle(img, CONTROL_CIRCLE_XY_CENTER,
               CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

    # TODO: bounds validation for both hands (ignore hands outside of certain ranges)
    if lh_data:
        # must normalize. if a hand has center of (0.5, 0.5) it means the hand is in the middle of the screen, but our xy control center is
        # at (0.25, 0.5) so we need to scale the x value by 2
        xy_ctl_x_pct_normalized = min((lh_data.center_perc[0] - 0.25) * 2, 1.0)
        xy_ctl_x = int(
            xy_ctl_x_pct_normalized * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[0]

        xy_ctl_y_pct_normalized = min((lh_data.center_perc[1] - 0.5) * 2, 1.0)
        xy_ctl_y = int(
            xy_ctl_y_pct_normalized
            * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[1]
        cv2.circle(img, (xy_ctl_x, xy_ctl_y),
                   4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    cv2.circle(img, CONTROL_CIRCLE_Z_APERATURE_CENTER,
               CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)
    if rh_data:
        z_ctl_pct_normalized = min(
            (rh_data.center_perc[1] - 0.50) * 2, 1.0)
        # y being the height in the image
        z_ctl_y = int(
            z_ctl_pct_normalized * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[1]

        aperature_ctl_x_pct_normalized = min(
            (rh_data.center_perc[0] - 0.75) * 2, 1.0)
        # x being the width in the image
        aperature_ctl_x = int(
            aperature_ctl_x_pct_normalized * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[0]
        print(f"apx, py: {aperature_ctl_x}, {z_ctl_y}")
        cv2.circle(img, (aperature_ctl_x, z_ctl_y),
                   4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    cv2.line(img, (int(CAMERA_WIDTH/2), 0),
             (int(CAMERA_WIDTH/2), CAMERA_HEIGHT), CV_DRAW_COLOR_PRIMARY, 1)


def main():
    cap = cv2.VideoCapture(sys.argv[1] if len(sys.argv) > 1 else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    pTime = time.monotonic()

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_reduced = cv2.resize(img_rgb, (0, 0), fx=0.25, fy=0.25)

        hand_res = hands.process(img_rgb_reduced)
        handcount = 0
        lh_data, rh_data = None, None
        if hand_res.multi_hand_landmarks:
            handcount = len(hand_res.multi_hand_landmarks)
            for (handedness, hand_landmarks) in zip(hand_res.multi_handedness, hand_res.multi_hand_landmarks):
                if handedness.classification[0].label.lower() == 'left':
                    lh_data = HandData(handedness, hand_landmarks.landmark)
                else:
                    rh_data = HandData(handedness, hand_landmarks.landmark)

                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    mpDraw.draw_landmarks(
                        img, hand_landmarks, mpHands.HAND_CONNECTIONS)

        cTime = time.monotonic()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        draw_control_overlay(img, lh_data, rh_data)
        cv2.imshow("Test", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
