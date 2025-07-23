import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort.sort import *
from utilts import get_car, read_license_plate, write_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#load model
coco_model = YOLO('model/finetune_v8_vehicles.pt').to(device)
license_plate_model = YOLO('model/yolov8n_fine_tune_number_plate.pt').to(device)


mot_tracker = Sort()

#load video
path_video = "./test/car.mp4"
cap = cv2.VideoCapture(path_video)


vehicles = [2, 3, 5, 7]
num_frames = -1
results = {}
ret = True
while ret  :
    num_frames +=1
    ret, frame = cap.read()
    if ret :
        results[num_frames] = {}
        #detect vehicles
        with torch.no_grad():
            detection_vehs = coco_model(frame)[0]
        detection_vehs_ = []
        for detect in detection_vehs.boxes.data.tolist():
            x1, y1, x2, y2, score, id_class = detect
            if int(id_class) in vehicles:
                    detection_vehs_.append([x1, y1, x2, y2, score])


        #track vehicles
        track_id = mot_tracker.update(np.asarray(detection_vehs_))

        #detec license plate
        with torch.no_grad():
            license_plates = license_plate_model.predict(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, id_class = license_plate


            #assign license plate to car
            x_car1, y_car1, x_car2, y_car2, car_Id = get_car(license_plate, track_id)

            if car_Id != -1:
                #crop license
                license_plates_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                #process
                license_plates_crop_gray = cv2.cvtColor(license_plates_crop, cv2.COLOR_BGR2GRAY)
                _, license_plates_crop_thresh = cv2.threshold(license_plates_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # cv2.imshow("crop", license_plates_crop)
                # cv2.imshow("thresh", license_plates_crop_thresh)
                # cv2.waitKey(0)

                #read number of license
                license_plates_text, license_plates_text_score = read_license_plate(license_plates_crop_thresh)

                if license_plates_text is not None:
                    results[num_frames][car_Id] = {'car': {'bbox': [x_car1, y_car1, x_car2, y_car2] },
                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                     'text': license_plates_text,
                                                                     'bbox_score': score,
                                                                    'text_score': license_plates_text_score}}


# write results
write_csv(results, "./car.csv")


