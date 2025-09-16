import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Finger indices (tip landmarks)
finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Flip camera for mirror effect
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect Left or Right hand
            label = hand_handedness.classification[0].label  # "Left" or "Right"

            fingers = []

            # Thumb logic depends on hand
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

            # Other 4 fingers (same logic for both hands)
            for tip_id in finger_tips[1:]:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    fingers.append(1)  # finger up
                else:
                    fingers.append(0)  # finger down

            # Detect gestures
            gesture = "Unknown"
            if fingers == [1, 0, 0, 0, 0]:
                gesture = "👍 Thumbs Up"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "✌️ Peace"
            elif fingers == [0, 0, 0, 0, 0]:
                gesture = "👊 Fist"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "🖐 Open Palm"

            cv2.putText(img, f"{label} Hand: {gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Gesture Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
