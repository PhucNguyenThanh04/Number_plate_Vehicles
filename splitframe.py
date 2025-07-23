import cv2

# Đường dẫn tới video
video_path = "./test/car.mp4"

# Frame bạn muốn trích xuất (ví dụ: frame thứ 100)
target_frame_index = 100

# Mở video
cap = cv2.VideoCapture(video_path)

# Kiểm tra mở thành công
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Di chuyển tới frame mong muốn
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)

# Đọc frame
ret, frame = cap.read()

if ret:
    # Lưu ảnh ra file (JPEG hoặc PNG)
    cv2.imwrite("frame_100.jpg", frame)
    print("Đã lưu frame thành công.")
else:
    print("Không thể đọc frame.")

# Giải phóng tài nguyên
cap.release()
