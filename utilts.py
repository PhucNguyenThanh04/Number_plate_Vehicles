import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}



# def write_csv(results, output_path):
#
#     with open(output_path, 'w') as f:
#         f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
#                                                 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
#                                                 'license_number_score'))
#
#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 print(results[frame_nmr][car_id])
#                 if 'car' in results[frame_nmr][car_id].keys() and \
#                    'license_plate' in results[frame_nmr][car_id].keys() and \
#                    'text' in results[frame_nmr][car_id]['license_plate'].keys():
#                     f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
#                                                             car_id,
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['car']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][3]),
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][3]),
#                                                             results[frame_nmr][car_id]['license_plate']['bbox_score'],
#                                                             results[frame_nmr][car_id]['license_plate']['text'],
#                                                             results[frame_nmr][car_id]['license_plate']['text_score'])
#                             )
#         f.close()
def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'))

        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                car_data = results[frame_nmr][car_id]

                # Dữ liệu mặc định là None nếu không có
                car_bbox = '[{} {} {} {}]'.format(*car_data['car']['bbox']) if 'car' in car_data else None
                lp_bbox = '[{} {} {} {}]'.format(*car_data['license_plate']['bbox']) if 'license_plate' in car_data else None
                lp_score = car_data['license_plate'].get('bbox_score') if 'license_plate' in car_data else None
                lp_text = car_data['license_plate'].get('text') if 'license_plate' in car_data else None
                lp_text_score = car_data['license_plate'].get('text_score') if 'license_plate' in car_data else None

                f.write('{},{},{},{},{},{},{}\n'.format(
                    frame_nmr,
                    car_id,
                    car_bbox,
                    lp_bbox,
                    lp_score,
                    lp_text,
                    lp_text_score
                ))



def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score
    return None, None


def get_car(license_plate, vehicle_track_ids):
    foundIt = False
    x1, y1, x2, y2, score, id_class = license_plate
    for j in range(len(vehicle_track_ids)):
        x_car1, y_car1, x_car2, y_car2, car_Id  = vehicle_track_ids[j]

        if x1> x_car1 and y1 > y_car1 and x2 < x_car2 and y2 < y_car2:
            car_idx = j
            foundIt = True
            break
    if foundIt:
        return vehicle_track_ids[car_idx]
    return -1, -1, -1, -1, -1



