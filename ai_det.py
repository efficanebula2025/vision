from ultralytics import YOLO
import cv2
import os

# ----- CONFIG -----
MODEL = "yolov8n.pt"     # pretrained COCO model
TARGET_CLASSES = {"keyboard", "mouse", "chair","toothbrush"}
CONF_THRESH = 0.25
SOURCE = 0                # 0 = webcam; or "images/" for folder
OUT_DIR = "outputs"
# ------------------

os.makedirs(OUT_DIR, exist_ok=True)

# load model and force to CPU
model = YOLO(MODEL)
model.to("cpu")   # ensures everything runs on CPU

def process_frame(frame, frame_id=0):
    # always run on CPU
    results = model(frame, conf=CONF_THRESH, device="cpu")
    annotated = frame.copy()

    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            name = model.names.get(cls_id, str(cls_id))
            conf = float(boxes.conf[i].item())
            if name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            label = f"{name} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, results

def run_from_camera(source=0):
    cap = cv2.VideoCapture(source)
    idx = 0
    if not cap.isOpened():
        print("Unable to open video source:", source)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = process_frame(frame, idx)
        cv2.imshow("YOLO detect (CPU mode, press q to quit)", annotated)

        if idx % 30 == 0:
            fn = os.path.join(OUT_DIR, f"frame_{idx:06d}.jpg")
            cv2.imwrite(fn, annotated)

        idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_from_camera(SOURCE)
