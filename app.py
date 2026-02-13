import cv2
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import queue

# ✅ ONLY NEW AUDIO IMPORTS (Guaranteed Fix)
from gtts import gTTS
from playsound import playsound
import os
import threading
import uuid

app = Flask(__name__)

# ================= AUDIO SYSTEM =================
speech_queue = queue.Queue()

last_spoken = None
last_time = 0
last_rep_audio_time = 0

SPEECH_COOLDOWN = 1.0
REP_LOCK_TIME = 0.8

LAST_AUDIO_TIME = 0
AUDIO_DELAY = 1.2


def speak(text, priority=False):
    global last_spoken, last_time, last_rep_audio_time
    if not text:
        return

    now = time.time()

    if not priority and (now - last_rep_audio_time) < REP_LOCK_TIME:
        return

    if priority:
        last_rep_audio_time = now
        speech_queue.put(text)
        return

    if text == last_spoken and (now - last_time) < SPEECH_COOLDOWN:
        return

    last_spoken = text
    last_time = now
    speech_queue.put(text)


# ✅ GUARANTEED AUDIO PLAYER THREAD (gTTS)
def audio_loop_thread():
    while True:
        if not speech_queue.empty():
            text = speech_queue.get()

            try:
                filename = f"audio_{uuid.uuid4()}.mp3"

                tts = gTTS(text=text, lang="en")
                tts.save(filename)

                playsound(filename)

                os.remove(filename)

            except Exception as e:
                print("Audio Error:", e)

        time.sleep(0.05)


# ================= CONFIG =================
MODEL_PATH = r"C:\Users\Pc\OneDrive\Desktop\Gym_trainer2_pushup\runs\pose\runs\pushup_pose\yolo_pose_train_val\weights\best.pt"

CONF = 0.25

ELBOW_DOWN_STD = 115
ELBOW_UP_STD = 150
ELBOW_DOWN_WIDE = 125
ELBOW_UP_WIDE = 145

BACK_STRAIGHT = 135
WIDE_RATIO = 1.3

MIN_REP_TIME = 0.4
MAX_REP_TIME = 4.0
MIN_PLANK_FRAMES = 3

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ================= STATE =================
state = {
    "total": 0,
    "standard": 0,
    "wide": 0,
    "situp": 0,
    "down": False,
    "rep_start": None,
    "times": [],
    "plank_frames": 0,
    "feedback": "Waiting for movement…",
    "situp_stage": "down"
}


# ================= GEOMETRY =================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


# ================= POSTURE CHECKS =================
def is_pushup_support_position(kpts):
    hip, knee, ankle = kpts[11], kpts[13], kpts[15]
    hip_knee_dist = abs(hip[1] - knee[1])
    knee_ankle_dist = abs(knee[1] - ankle[1])
    return hip_knee_dist >= knee_ankle_dist * 0.35


def is_valid_pushup_frame(kpts, h):
    shoulder, hip, ankle = kpts[5], kpts[11], kpts[15]
    back_angle = calculate_angle(shoulder, hip, ankle)

    if not is_pushup_support_position(kpts):
        return False, "Get into position as per the instruction given"

    if abs(shoulder[1] - hip[1]) > 120:
        return False, "Align body straighter"

    if back_angle < 120:
        return False, "Keep back straighter"

    return True, "Good posture"


# ================= PUSH-UP TYPE =================
def detect_pushup_type(kpts):
    shoulder_width = np.linalg.norm(kpts[5] - kpts[6])
    wrist_width = np.linalg.norm(kpts[9] - kpts[10])

    if wrist_width > shoulder_width * WIDE_RATIO:
        return "Wide", ELBOW_DOWN_WIDE, ELBOW_UP_WIDE

    return "Standard", ELBOW_DOWN_STD, ELBOW_UP_STD


# ================= SIT-UP CHECK =================
def is_situp_position(kpts):
    hip = kpts[11]
    knee = kpts[13]
    return abs(hip[1] - knee[1]) < 50


# ================= COUNTER =================
def update_counter(kpts, h):

    if is_situp_position(kpts):

        SHOULDER, HIP = kpts[5], kpts[11]
        torso_lift = HIP[1] - SHOULDER[1]

        SITUP_DOWN_DIST = 50
        SITUP_UP_DIST = 110

        if state["situp_stage"] == "down" and torso_lift > SITUP_UP_DIST:
            state["situp_stage"] = "up"
            return "Sit-up up"

        elif state["situp_stage"] == "up" and torso_lift < SITUP_DOWN_DIST:
            state["situp_stage"] = "down"
            state["situp"] += 1
            state["total"] += 1
            return "Sit-up rep counted"

    shoulder, elbow, wrist = kpts[5], kpts[7], kpts[9]
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    valid, msg = is_valid_pushup_frame(kpts, h)

    if not valid:
        state["down"] = False
        state["plank_frames"] = 0
        return msg

    state["plank_frames"] += 1

    if state["plank_frames"] < MIN_PLANK_FRAMES:
        return "Hold position"

    ptype, down_thr, up_thr = detect_pushup_type(kpts)

    if elbow_angle < down_thr and not state["down"]:
        state["down"] = True
        state["rep_start"] = time.time()
        return f"{ptype} down"

    if elbow_angle > up_thr and state["down"]:
        rep_time = time.time() - state["rep_start"]
        state["down"] = False

        if MIN_REP_TIME < rep_time < MAX_REP_TIME:
            state["total"] += 1
            state["times"].append(rep_time)

            if ptype == "Standard":
                state["standard"] += 1
            else:
                state["wide"] += 1

            return f"{ptype} rep counted"

        return "Too fast or slow"

    return f"{ptype} moving"


# ================= VIDEO STREAM =================
def generate_frames():
    global LAST_AUDIO_TIME

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        h = frame.shape[0]
        results = model(frame, conf=CONF, verbose=False)

        if results and results[0].keypoints is not None:
            kpts_all = results[0].keypoints.xy.cpu().numpy()

            if len(kpts_all):
                feedback = update_counter(kpts_all[0], h)
                state["feedback"] = feedback

                now = time.time()

                if now - LAST_AUDIO_TIME > AUDIO_DELAY:

                    if "rep counted" in feedback:
                        speak(f"Rep {state['total']}", priority=True)
                    else:
                        speak(feedback)

                    LAST_AUDIO_TIME = now

                frame = results[0].plot(img=frame)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


# ================= WEEKLY PLAN =================
def generate_weekly_plan(age, weight):
    base_pushups = 10
    base_situps = 15

    if age < 25:
        age_factor = 1.3
    elif age < 40:
        age_factor = 1.0
    else:
        age_factor = 0.7

    if weight < 60:
        weight_factor = 1.2
    elif weight < 80:
        weight_factor = 1.0
    else:
        weight_factor = 0.8

    pushup_start = int(base_pushups * age_factor * weight_factor)
    situp_start = int(base_situps * age_factor * weight_factor)

    pushup_plan = [pushup_start + i * 2 for i in range(7)]
    situp_plan = [situp_start + i * 3 for i in range(7)]

    return pushup_plan, situp_plan


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/plan", methods=["POST"])
def plan():
    age = int(request.form["age"])
    weight = int(request.form["weight"])

    pushup_plan, situp_plan = generate_weekly_plan(age, weight)

    return render_template(
        "index.html",
        weekly_pushup_plan=pushup_plan,
        weekly_situp_plan=situp_plan
    )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/metrics")
def metrics():
    avg_time = round(np.mean(state["times"]), 2) if state["times"] else 0

    return jsonify({
        "total_reps": state["total"],
        "standard_reps": state["standard"],
        "wide_reps": state["wide"],
        "situp_reps": state["situp"],
        "avg_time": avg_time,
        "feedback": state["feedback"]
    })


# ================= RUN =================
if __name__ == "__main__":

    # ✅ Start guaranteed audio thread
    threading.Thread(target=audio_loop_thread, daemon=True).start()

    speak("Workout started", priority=True)
    app.run(debug=True)
