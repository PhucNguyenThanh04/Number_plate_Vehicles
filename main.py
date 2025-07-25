import ast
import cv2
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from detect import writer_bbx
from add_missing_data import load_write, interpolate_bounding_boxes



def get_args():
    parser = ArgumentParser(description= "visualize number plate recognition")
    parser.add_argument("--input_video", "-v", type=str, required=True, help="path video detect")
    parser.add_argument("--csv", "-c", type=str, required=True, help="path file csv detected")
    parser.add_argument("--csv_full", "-f", type=str, required=True, help="path file csv full")
    parser.add_argument("--out_video", "-o", type=str, default= "./out", help="path file csv full")
    args = parser.parse_args()
    return args


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def main():
    args = get_args()
    # Load dữ liệu từ file CSV
    cap = cv2.VideoCapture(args.input_video)

    writer_bbx(cap, args.csv)
    print("xong main")

    load_write(args.csv, args.csv_full)
    print("add ")

    results = pd.read_csv(args.csv)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        filtered = results[results['car_id'] == car_id]
        if filtered.empty:
            continue

        max_ = filtered['license_number_score'].max()
        best_rows = filtered[filtered['license_number_score'] == max_]

        if best_rows.empty:
            license_plate[car_id] = {'license_crop': None, 'license_plate_number': "UNKNOWN"}
            continue

        try:
            license_plate_number = best_rows['license_number'].iloc[0]
            license_plate[car_id] = {'license_crop': None, 'license_plate_number': license_plate_number}

            # Lấy frame và crop license plate
            frame_index = best_rows['frame_nmr'].iloc[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            bbox_str = best_rows['license_plate_bbox'].iloc[0]
            x1, y1, x2, y2 = ast.literal_eval(
                bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )

            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
            license_plate[car_id]['license_crop'] = license_crop

        except Exception as e:
            print(f"Error processing car_id={car_id}: {e}")
            license_plate[car_id] = {'license_crop': None, 'license_plate_number': "UNKNOWN"}

    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                    df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(
                        ' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                # draw license plate
                x1, y1, x2, y2 = ast.literal_eval(
                    df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ',
                                                                                                            ' ').replace(
                        ' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                if license_crop is not None:
                    H, W, _ = license_crop.shape

                try:
                    frame[int(car_y1) - H - 100:int(car_y1) - 100,
                    int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                    int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        17)

                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)

                except:
                    pass

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out.release()
    cap.release()

if __name__ == '__main__':
    main()