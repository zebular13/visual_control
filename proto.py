import sys
import time

import cv2
import mediapipe as mp


def main():
    cap = cv2.VideoCapture(sys.argv[1] if len(sys.argv) > 1 else 0)
    pTime = time.monotonic()

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hand_res = hands.process(imgRGB)
        if hand_res.multi_hand_landmarks:
            for hand_landmarks in hand_res.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
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
        cv2.imshow("Test", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
