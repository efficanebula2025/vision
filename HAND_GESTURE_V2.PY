import os
import cv2
import mediapipe as mp

# Suppress TensorFlow Lite warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Finger tip landmarks
finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# Open camera
cap = cv2.VideoCapture(2)  # Try (1) if 0 doesn't work

if not cap.isOpened():
    print("❌ Could not open camera. Try changing the index in VideoCapture().")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Ignoring empty camera frame.")
        continue

    # Flip camera for mirror effect
    img = cv2.flip(img, 1)

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Hand label
            label = hand_handedness.classification[0].label  # "Left" or "Right"
            fingers = []

            # Thumb logic
            if label == "Right":
                if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:  # Left hand (mirrored)
                if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_tips[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Other fingers
            for tip_id in finger_tips[1:]:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    fingers.append(1)  # finger open
                else:
                    fingers.append(0)  # finger closed

            # Gesture detection
            gesture = "Unknown"
            if fingers == [1, 0, 0, 0, 0]:
                gesture = "👍 Thumbs Up"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "✌️ Peace"
            elif fingers == [0, 0, 0, 0, 0]:
                gesture = "👊 Fist"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "🖐 Open Palm"

            # Show result text
            cv2.putText(img, f"{label} Hand: {gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show camera feed
    cv2.imshow("Hand Gesture Detection", img)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release camera
cap.release()
cv2.destroyAllWindows()
