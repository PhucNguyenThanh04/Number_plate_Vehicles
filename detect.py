from tqdm import tqdm  # thêm vào đầu file
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import *
from utilts import get_car, read_license_plate, write_csv


def writer_bbx(cap , csv_output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    coco_model = YOLO('model/finetune_v8_vehicles.pt').to(device)
    license_plate_model = YOLO('model/yolov8n_fine_tune_number_plate.pt').to(device)

    mot_tracker = Sort()

    vehicles = [2, 3, 5, 7]
    num_frames = -1
    results = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 👈 lấy tổng số frame
    pbar = tqdm(total=total_frames, desc="Processing video", colour="cyan")  # 👈 tạo tqdm progress bar

    ret = True
    while ret:
        num_frames += 1
        ret, frame = cap.read()
        if not ret:
            break

        results[num_frames] = {}

        # detect vehicles
        with torch.no_grad():
            detection_vehs = coco_model(frame, verbose= False)[0]

        detection_vehs_ = []
        for detect in detection_vehs.boxes.data.tolist():
            x1, y1, x2, y2, score, id_class = detect
            if int(id_class) in vehicles:
                detection_vehs_.append([x1, y1, x2, y2, score])

        # track vehicles
        if detection_vehs_:
            track_id = mot_tracker.update(np.asarray(detection_vehs_))
        else:
            track_id = []
        # detect license plate
        with torch.no_grad():
            license_plates = license_plate_model.predict(frame, verbose = False)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, id_class = license_plate

            x_car1, y_car1, x_car2, y_car2, car_Id = get_car(license_plate, track_id)

            if car_Id != -1:
                license_plates_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plates_crop_gray = cv2.cvtColor(license_plates_crop, cv2.COLOR_BGR2GRAY)
                _, license_plates_crop_thresh = cv2.threshold(license_plates_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plates_text, license_plates_text_score = read_license_plate(license_plates_crop_thresh)

                results[num_frames][car_Id] = {
                    'car': {'bbox': [x_car1, y_car1, x_car2, y_car2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plates_text,
                        'bbox_score': score,
                        'text_score': license_plates_text_score
                    }
                }
        pbar.update(1)
    pbar.close()
    write_csv(results, csv_output_path)
