import cv2
import time
import numpy as np
from ultralytics import YOLO
import pyttsx3
import queue

# ================= AUDIO SYSTEM =================
speech_queue = queue.Queue()

last_spoken = None
last_time = 0
last_rep_audio_time = 0

SPEECH_COOLDOWN = 1.0
REP_LOCK_TIME = 0.8

engine = pyttsx3.init()
engine.setProperty("rate", 190)
engine.setProperty("volume", 1.0)
engine.startLoop(False)  # non-blocking


def speak(text, priority=False):
    global last_spoken, last_time, last_rep_audio_time
    if not text:
        return

    now = time.time()

    # block feedback if rep was just spoken
    if not priority and (now - last_rep_audio_time) < REP_LOCK_TIME:
        return

    # rep counting always allowed
    if priority:
        last_rep_audio_time = now
        speech_queue.put(text)
        return

    # cooldown for normal feedback
    if text == last_spoken and (now - last_time) < SPEECH_COOLDOWN:
        return

    last_spoken = text
    last_time = now
    speech_queue.put(text)


def audio_loop():
    if not speech_queue.empty():
        engine.say(speech_queue.get())
    engine.iterate()


# ================= CONFIG =================
MODEL_PATH = r"C:\Users\Pc\OneDrive\Desktop\Gym_trainer2_pushup\runs\pose\runs\pushup_pose\yolo_pose_train_val\weights\best.pt"
VIDEO_PATH = r"C:\Users\Pc\Downloads\Exercise_Tracker\Crush_Your_Workout_in_Minutes_Quick_Effective_Routine_720P (2).mp4"

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


# ================= GEOMETRY =================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


# ================= POSTURE =================
def is_pushup_support_position(kpts):
    hip, knee, ankle = kpts[11], kpts[13], kpts[15]

    hip_knee_dist = abs(hip[1] - knee[1])
    knee_ankle_dist = abs(knee[1] - ankle[1])

    if hip_knee_dist < knee_ankle_dist * 0.35:
        return False
    return True


def is_horizontal(shoulder, hip):
    return abs(shoulder[1] - hip[1]) < 80  


def is_low_position(shoulder, h):
    return shoulder[1] > h * 0.35 


def is_plank(back_angle):
    return back_angle >= BACK_STRAIGHT 


def is_valid_pushup_frame(kpts, h):
    shoulder, hip, ankle = kpts[5], kpts[11], kpts[15]
    back_angle = calculate_angle(shoulder, hip, ankle)

    if not is_pushup_support_position(kpts):
        return False, "Get into  position"

    if not is_horizontal(shoulder, hip):
        return False, "Align body"

    if not is_low_position(shoulder, h):
        return False, "Go lower"

    if not is_plank(back_angle):
        return False, "Keep back straight"

    return True, "Good posture"


# ================= SIT-UP POSITION CHECK =================
def is_situp_position(kpts):
    hip = kpts[11]
    knee = kpts[13]

    # Situp posture → knee is close to hip vertically
    return abs(hip[1] - knee[1]) < 50


# ================= PUSH-UP TYPE =================
def detect_pushup_type(kpts):
    shoulder_width = np.linalg.norm(kpts[5] - kpts[6])
    wrist_width = np.linalg.norm(kpts[9] - kpts[10])

    if wrist_width > shoulder_width * WIDE_RATIO:
        return "Wide", ELBOW_DOWN_WIDE, ELBOW_UP_WIDE

    return "Standard", ELBOW_DOWN_STD, ELBOW_UP_STD

# ================= Counter =================
def update_pushup_counter(kpts, h, state):

    feedback = ""

    # ✅ Detect if user is doing sit-up posture
    doing_situp = is_situp_position(kpts)

    # ================= PUSH-UP =================
    if not doing_situp:

        shoulder, elbow, wrist = kpts[5], kpts[7], kpts[9]
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        valid, msg = is_valid_pushup_frame(kpts, h)

        if not valid:
            state["down"] = False
            state["plank_frames"] = 0
            feedback = msg

        else:
            state["plank_frames"] += 1

            if state["plank_frames"] < MIN_PLANK_FRAMES:
                feedback = "Hold position"

            else:
                ptype, down_thr, up_thr = detect_pushup_type(kpts)

                # Down position
                if elbow_angle < down_thr and not state["down"]:
                    state["down"] = True
                    state["rep_start"] = time.time()
                    feedback = f"{ptype} down"

                # Up position = REP COUNT
                if elbow_angle > up_thr and state["down"]:
                    rep_time = time.time() - state["rep_start"]
                    state["down"] = False

                    if MIN_REP_TIME < rep_time < MAX_REP_TIME:
                        state["total"] += 1

                        if ptype == "Standard":
                            state["standard"] += 1
                        else:
                            state["wide"] += 1

                        state["times"].append(rep_time)
                        feedback = f"{ptype} rep counted"

                    else:
                        feedback = "Too fast"

    # ================= SIT-UP =================
    SHOULDER, HIP = kpts[5], kpts[11]
    torso_lift = HIP[1] - SHOULDER[1]

    SITUP_DOWN_DIST = 40
    SITUP_UP_DIST = 120

    # ✅ Only detect situps if person is actually in situp posture
    if doing_situp:

        if "situp_stage" not in state:
            state["situp_stage"] = "down"

        # Down → Up
        if state["situp_stage"] == "down" and torso_lift > SITUP_UP_DIST:
            state["situp_stage"] = "up"
            feedback = "Sit-up up"

        # Up → Down = REP COUNT
        elif state["situp_stage"] == "up" and torso_lift < SITUP_DOWN_DIST:
            state["situp_stage"] = "down"

            state["situp"] += 1
            state["total"] += 1

            feedback = "Sit-up rep counted"

    return feedback


# ================= MAIN =================
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    state = {
        "total": 0,
        "standard": 0,
        "wide": 0,
        "situp": 0,
        "down": False,
        "rep_start": None,
        "plank_frames": 0,
        "times": []
    }

    speak("Start workout", priority=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h = frame.shape[0]
        results = model(frame, conf=CONF, verbose=False)

        annotated = frame.copy()
        feedback = ""

        if results and results[0].keypoints is not None:
            kpts_all = results[0].keypoints.xy.cpu().numpy()

            if len(kpts_all):
                feedback = update_pushup_counter(kpts_all[0], h, state)
                annotated = results[0].plot(img=annotated)

                if "rep counted" in feedback:
                    speak(f"Rep {state['total']}", priority=True)
                else:
                    speak(feedback)

        audio_loop()

        avg_time = np.mean(state["times"]) if state["times"] else 0

        # ================= OVERLAY =================
        cv2.putText(annotated, f"Total: {state['total']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(annotated, f"Standard: {state['standard']}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.putText(annotated, f"Wide: {state['wide']}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.putText(annotated, f"Sit-ups: {state['situp']}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.putText(annotated, f"Avg Rep Time: {avg_time:.2f}s", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.putText(annotated, f"Feedback: {feedback}", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

        # Resize display only
        display_width = 640
        display_height = int(annotated.shape[0] * (display_width / annotated.shape[1]))
        resized_frame = cv2.resize(annotated, (display_width, display_height))

        cv2.imshow("AI Trainer", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    try:
        engine.stop()
    except:
        pass


if __name__ == "__main__":
    main()
