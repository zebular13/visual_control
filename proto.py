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
        if hand_res.multi_hand_landmarks:
            for (handedness, hand_landmarks) in zip(hand_res.multi_handedness, hand_res.multi_hand_landmarks):
                print(handedness.classification[0].label)
                for id, lm in enumerate(hand_landmarks.landmark):
                    # print(id)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(
                    img, hand_landmarks, mpHands.HAND_CONNECTIONS)

        cTime = time.monotonic()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # control overlay
        cv2.circle(img, CONTROL_CIRCLE_XY_CENTER,
                   CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)
        cv2.circle(img, CONTROL_CIRCLE_Z_APERATURE_CENTER,
                   CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

        cv2.line(img, (int(CAMERA_WIDTH/2), 0),
                 (int(CAMERA_WIDTH/2), CAMERA_HEIGHT), CV_DRAW_COLOR_PRIMARY, 1)

        cv2.imshow("Test", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
