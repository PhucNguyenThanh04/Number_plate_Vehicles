import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import *
from utilts import get_car, read_license_plate

# Hàm vẽ khung bao quanh xe (đơn giản hóa)
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=2):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# Khởi tạo thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tải mô hình
model = YOLO('model/finetune_v8_vehicles.pt').to(device)

# Khởi tạo tracker
mot_tracker = Sort()

# Tải video
path_video = "./test/car.mp4"
cap = cv2.VideoCapture(path_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Danh sách class ID cho phương tiện
vehicles = [0, 1, 2, 3, 4, 6, 7]
plate_class = 5

# Lưu trữ thông tin biển số
license_plate = {}
frame_skip = 2  # Xử lý mỗi 2 khung hình
frame_count = 0

# Xử lý video
ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Bỏ qua khung hình

    # Giảm độ phân giải
    frame = cv2.resize(frame, (640, 360))
    display_frame = frame.copy()

    # Phát hiện phương tiện và biển số
    with torch.no_grad():
        detections = model(frame)[0]  # Tăng ngưỡng confidence để giảm phát hiện sai

    # Tách phương tiện và biển số
    detection_vehicles = []
    detection_plates = []
    for detect in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, id_class = detect
        id_class = int(id_class)
        if id_class in vehicles:
            detection_vehicles.append([x1, y1, x2, y2, score])
        elif id_class == plate_class:
            detection_plates.append([x1, y1, x2, y2, score, id_class])

    # Theo dõi phương tiện
    track_ids = mot_tracker.update(np.asarray(detection_vehicles))

    # Xử lý biển số
    for license_plate_data in detection_plates:
        x1, y1, x2, y2, score, id_class = license_plate_data

        # Gán biển số cho xe
        x_car1, y_car1, x_car2, y_car2, car_id = get_car(license_plate_data, track_ids)

        if car_id != -1:
            # Chỉ chạy OCR nếu chưa có biển số hoặc điểm tin cậy thấp
            if car_id not in license_plate or license_plate[car_id]['license_number_score'] < 0.7:
                license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_crop = cv2.resize(license_crop, (int((x2 - x1) * 200 / (y2 - y1)), 200))  # Giảm kích thước

                # Xử lý ảnh biển số
                license_crop_gray = cv2.cvtColor(license_crop, cv2.COLOR_BGR2GRAY)
                license_crop_gray = cv2.equalizeHist(license_crop_gray)  # Cân bằng histogram
                _, license_crop_thresh = cv2.threshold(license_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Đọc văn bản biển số
                license_text, license_text_score = read_license_plate(license_crop_thresh)

                if license_text is not None:
                    license_plate[car_id] = {
                        'license_crop': license_crop,
                        'license_plate_number': license_text,
                        'license_number_score': license_text_score
                    }

            # Vẽ khung bao xe
            draw_border(display_frame, (int(x_car1), int(y_car1)), (int(x_car2), int(y_car2)), (0, 255, 0), 5)

            # Vẽ khung bao biển số
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            # Hiển thị văn bản biển số (bỏ hiển thị ảnh để tăng tốc)
            if car_id in license_plate:
                license_text = license_plate[car_id]['license_plate_number']
                (text_width, text_height), _ = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cv2.putText(display_frame,
                            license_text,
                            (int((x_car2 + x_car1 - text_width) / 2), int(y_car1 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 0),
                            3)

    # Hiển thị khung hình
    cv2.imshow('Real-time License Plate Detection', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()