import os
import cv2
import mediapipe as mp
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pyttsx3
import requests
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize text-to-speech
engine = pyttsx3.init()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
pushup_count = 0
pushup_stage = None

# ThingSpeak details
THINGSPEAK_API_KEY = "BB8FBKH48AQCLF16"
THINGSPEAK_CHANNEL_ID = "2681631"
THINGSPEAK_URL = f"https://api.thingspeak.com/update?api_key={THINGSPEAK_API_KEY}"

# Email credentials
GMAIL_USER = "premjayapaulpidathala@gmail.com"
GMAIL_PASSWORD = "elvc pmmq jhab vyzf"
RECEIVER_EMAIL = "karasanisrisivakotireddy@gmail.com"


# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# Send data to ThingSpeak
def send_to_thingspeak(count):
    try:
        response = requests.get(THINGSPEAK_URL, params={'field1': count})
        if response.status_code == 200:
            print(f"Push-up count {count} sent to ThingSpeak")
        else:
            print("ThingSpeak upload failed")
    except Exception as e:
        print("ThingSpeak Error:", e)


# Send email summary
def send_email(count):
    if count >= 1:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "Push-Up Count Summary"

        body = f"You have completed {count} push-ups."
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.sendmail(GMAIL_USER, RECEIVER_EMAIL, msg.as_string())
        server.quit()

        print("Email sent successfully")


# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        if frame_count % 5 == 0:
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                landmarks = results.pose_landmarks.landmark

                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                ]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Push-up logic
                if angle > 160:
                    pushup_stage = "up"
                elif angle < 70 and pushup_stage == "up":
                    pushup_stage = "down"
                    pushup_count += 1

                    engine.say(f"Push-up number {pushup_count}")
                    engine.runAndWait()

                    send_to_thingspeak(pushup_count)

                cv2.putText(frame, f"Angle: {int(angle)}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Push-ups: {pushup_count}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Push-up Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted")

finally:
    send_email(pushup_count)
    cap.release()
    cv2.destroyAllWindows()
