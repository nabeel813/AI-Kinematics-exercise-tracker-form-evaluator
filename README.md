# AI-Kinematics-exercise-tracker-form-evaluator
AI-Kinematics-Exercise-Tracker-Form-Evaluator
AI Kinematics Trainer :
Real-Time Exercise Tracker & Form Evaluator using YOLOv8 Pose Estimation
AI Kinematics Trainer is a real-time fitness assistant that detects human body posture using YOLOv8 Pose, counts repetitions, evaluates form, and provides live feedback with audio coaching.

This project focuses on Push-ups (Standard & Wide) and Sit-ups, with accurate rep counting and posture correction in real time.

Features
Real-time pose detection using YOLOv8 Pose Model
Exercise repetition counting

Standard Push-ups
Wide Push-ups
Sit-ups
Form Evaluation & Posture Feedback

Back alignment check
Correct push-up support position
Sit-up stage detection
Live Web Dashboard (Flask UI)

Total reps
Exercise-wise reps
Average rep time
Feedback display
Audio Feedback System

Rep announcements
Real-time coaching feedback using gTTS + playsound
Weekly Workout Plan Generator

User enters Age + Weight
Automatically generates 7-day push-up & sit-up plan
Demo Output
Live webcam stream with pose skeleton overlay
Rep counter updates instantly
Feedback displayed + spoken aloud
Weekly plan shown on the dashboard
---Flowchart

Tech Stack
| Component | Technology |

|----------|------------|

| Pose Estimation | YOLOv8 Pose (Ultralytics) |

| Backend | Flask (Python) |

| Frontend | HTML + CSS + JavaScript |

| Audio Feedback | gTTS + playsound |

| Computer Vision | OpenCV |

| Math & Angles | NumPy |

📂 Project Structure
AI-Kinematics-Trainer/

│

├── app.py                     # Main Flask + Pose + Rep Counter App

├── templates/

│   └── index.html             # Dashboard UI

│

├── static/

│   ├── style.css              # Styling

│   └── pushup.png             # Exercise reference image

│

├── weights/

│   └── best.pt                # Custom YOLOv8 Pose Model

│

└── README.md






About
No description, website, or topics provided.
Resources
 Readme
 Activity
Stars
 1 star
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Languages
Python
56.8%
 
HTML
28.5%
 
CSS
14.7%
Footer
