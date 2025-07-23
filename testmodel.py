# import cv2
# import torch
# from ultralytics import YOLO
#
# from main import device
#
# model = YOLO('model/finetune_v8_vehicles.pt').to(device)
# cap = cv2.VideoCapture("./test/car.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # with torch.no_grad():
#     results = model(frame)
#
#     annotated_frame = results.plot()  # Tự vẽ bbox, nhãn, confidence
#
#     cv2.imshow("YOLOv8 Detection", annotated_frame)
#     if cv2.waitKey(1)==27:
#         break
# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("./model/finetune_v8_vehicles.pt")  # hoặc yolov8n_custom.pt nếu bạn fine-tune

# Load video
cap = cv2.VideoCapture("test/car.mp4")  # đường dẫn video
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect object
    results = model(frame)[0]  # kết quả detect trên frame

    # Duyệt qua từng object detect được
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Hiển thị frame sau khi vẽ
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("YOLOv8 Detection", frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
