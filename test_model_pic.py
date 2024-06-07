import os
import cv2
from ultralytics import YOLO

# Assuming your image directory and model path
IMAGES_DIR = os.path.join('.', 'images')
image_path = os.path.join(IMAGES_DIR, r"C:\Users\USER\Desktop\thesis\code\MCD tests\test 1.jpg" )  # Replace 'your_image.jpg' with your image filename
output_dir = r"C:\Users\USER\Desktop\thesis\code\tests"
base_filename = os.path.basename(image_path)
new_filename = "{}_out.jpg".format(os.path.splitext(base_filename)[0])
image_path_out = os.path.join(output_dir, new_filename)
print(image_path_out) 
# Load your image
frame = cv2.imread(image_path)
H, W, _ = frame.shape
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
# Load the trained model
model_path = r"C:\Users\USER\Desktop\thesis\code\runs\detect\train5\weights\best.pt"
model = YOLO(model_path)
threshold = 0.5
# Perform detection
results = model(frame)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the annotated image
cv2.imwrite(image_path_out, frame)

cv2.destroyAllWindows()