import sys
import time

import cv2
import mediapipe


def main():
    cap = cv2.VideoCapture(sys.argv[1] if len(sys.argv) > 1 else 0)
    pTime = time.monotonic()

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cTime = time.monotonic()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Test", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
