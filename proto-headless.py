import cv2
import time
import threading
from flask import Flask, Response
import mediapipe as mp
from dataclasses import dataclass

# Configuration
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
DEVICE = '/dev/video2'
CV_DRAW_COLOR_PRIMARY = (255, 255, 0)
CONTROL_CIRCLE_DEADZONE_R = 50
CONTROL_CIRCLE_XY_CENTER = (CAMERA_WIDTH // 4, CAMERA_HEIGHT // 2)
CONTROL_CIRCLE_Z_APERATURE_CENTER = (3 * CAMERA_WIDTH // 4, CAMERA_HEIGHT // 2)

# Flask app
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

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

def draw_control_overlay(img, lh_data=None, rh_data=None):
    cv2.circle(img, CONTROL_CIRCLE_XY_CENTER, CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

    if lh_data:
        x_norm = min((lh_data.center_perc[0] - 0.25) * 4, 1.0)
        y_norm = min((lh_data.center_perc[1] - 0.5) * 2, 1.0)
        x = int(x_norm * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[0]
        y = int(y_norm * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_XY_CENTER[1]
        cv2.line(img, CONTROL_CIRCLE_XY_CENTER, (x, y), CV_DRAW_COLOR_PRIMARY, 1)
        cv2.circle(img, (x, y), 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    cv2.circle(img, CONTROL_CIRCLE_Z_APERATURE_CENTER, CONTROL_CIRCLE_DEADZONE_R, CV_DRAW_COLOR_PRIMARY, 2)

    if rh_data:
        z_norm = min((rh_data.center_perc[1] - 0.50) * 2, 1.0)
        x_norm = min((rh_data.center_perc[0] - 0.75) * 4, 1.0)
        x = int(x_norm * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[0]
        y = int(z_norm * CONTROL_CIRCLE_DEADZONE_R) + CONTROL_CIRCLE_Z_APERATURE_CENTER[1]
        cv2.line(img, CONTROL_CIRCLE_Z_APERATURE_CENTER, (x, y), CV_DRAW_COLOR_PRIMARY, 1)
        cv2.circle(img, (x, y), 4, CV_DRAW_COLOR_PRIMARY, cv2.FILLED)

    cv2.line(img, (CAMERA_WIDTH // 2, 0), (CAMERA_WIDTH // 2, CAMERA_HEIGHT), CV_DRAW_COLOR_PRIMARY, 1)

def process_frames():
    global latest_frame
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Failed to open video device.")
        return

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils

    pTime = time.monotonic()

    while True:
        ret, img = cap.read()
        if not ret:
            print("Frame grab failed")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_reduced = cv2.resize(img_rgb, (0, 0), fx=0.25, fy=0.25)

        results = hands.process(img_rgb_reduced)
        lh_data, rh_data = None, None

        if results.multi_hand_landmarks:
            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                if handedness.classification[0].label.lower() == 'left':
                    lh_data = HandData(handedness, hand_landmarks.landmark)
                else:
                    rh_data = HandData(handedness, hand_landmarks.landmark)
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

        cTime = time.monotonic()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        draw_control_overlay(img, lh_data, rh_data)

        with frame_lock:
            latest_frame = img.copy()

def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '<h1>MediaPipe Stream</h1><img src="/video_feed">'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start video thread
    t = threading.Thread(target=process_frames)
    t.daemon = True
    t.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)
