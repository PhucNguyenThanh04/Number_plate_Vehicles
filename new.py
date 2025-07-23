import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import *
from utilts import get_car, read_license_plate
import string

# Hàm vẽ khung bao quanh xe
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Khởi tạo thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tải mô hình
coco_model = YOLO('model/finetune_v8_vehicles.pt').to(device)
license_plate_model = YOLO('model/yolov8n_fine_tune_number_plate.pt').to(device)

# Khởi tạo tracker
mot_tracker = Sort()

# Tải video
path_video = "./test/car.mp4"
cap = cv2.VideoCapture(path_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Danh sách class ID cho phương tiện
vehicles = [2, 3, 5, 7]

# Lưu trữ thông tin biển số cho từng xe
license_plate = {}

# Xử lý video
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # Sao chép khung hình để hiển thị
        display_frame = frame.copy()

        # Phát hiện phương tiện
        with torch.no_grad():
            detection_vehs = coco_model(frame)[0]
        detection_vehs_ = []
        for detect in detection_vehs.boxes.data.tolist():
            x1, y1, x2, y2, score, id_class = detect
            if int(id_class) in vehicles:
                detection_vehs_.append([x1, y1, x2, y2, score])

        # Theo dõi phương tiện
        track_id = mot_tracker.update(np.asarray(detection_vehs_))

        # Phát hiện biển số
        with torch.no_grad():
            license_plates = license_plate_model.predict(frame)[0]
        for license_plate_data in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, id_class = license_plate_data

            # Gán biển số cho xe
            x_car1, y_car1, x_car2, y_car2, car_id = get_car(license_plate_data, track_id)

            if car_id != -1:
                # Cắt ảnh biển số
                license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

                # Xử lý ảnh biển số
                license_crop_gray = cv2.cvtColor(license_crop, cv2.COLOR_BGR2GRAY)
                _, license_crop_thresh = cv2.threshold(license_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Đọc văn bản biển số
                license_text, license_text_score = read_license_plate(license_crop_thresh)

                if license_text is not None:
                    # Lưu thông tin biển số
                    if car_id not in license_plate or license_text_score > license_plate.get(car_id, {}).get('license_number_score', 0):
                        license_plate[car_id] = {
                            'license_crop': license_crop,
                            'license_plate_number': license_text,
                            'license_number_score': license_text_score
                        }

                    # Vẽ khung bao xe
                    draw_border(display_frame, (int(x_car1), int(y_car1)), (int(x_car2), int(y_car2)), (0, 255, 0), 25)

                    # Vẽ khung bao biển số
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                    # Hiển thị ảnh biển số và văn bản
                    license_crop = license_plate[car_id]['license_crop']
                    H, W, _ = license_crop.shape
                    try:
                        display_frame[int(y_car1) - H - 100:int(y_car1) - 100,
                                      int((x_car2 + x_car1 - W) / 2):int((x_car2 + x_car1 + W) / 2), :] = license_crop

                        display_frame[int(y_car1) - H - 400:int(y_car1) - H - 100,
                                      int((x_car2 + x_car1 - W) / 2):int((x_car2 + x_car1 + W) / 2), :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate[car_id]['license_plate_number'],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            17)

                        cv2.putText(display_frame,
                                    license_plate[car_id]['license_plate_number'],
                                    (int((x_car2 + x_car1 - text_width) / 2), int(y_car1 - H - 250 + (text_height / 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    4.3,
                                    (0, 0, 0),
                                    17)
                    except:
                        pass

        # Giảm độ phân giải để hiển thị nhanh hơn
        display_frame = cv2.resize(display_frame, (1280, 720))

        # Hiển thị khung hình
        cv2.imshow('Real-time License Plate Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()