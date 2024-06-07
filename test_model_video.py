import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, r"C:\Users\USER\Desktop\test_video_2.mp4")  # Path to your video
output_dir = r"C:\Users\USER\Desktop\thesis\code\tests"
video_filename = os.path.basename(video_path)
video_path_out = os.path.join(output_dir, '{}_out.mp4'.format(os.path.splitext(video_filename)[0]))

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = r"C:\Users\USER\Desktop\thesis\code\runs\detect\train5\weights\best.pt"  
model = YOLO(model_path)

threshold = 0.5

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
