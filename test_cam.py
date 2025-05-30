from picamera2 import Picamera2
import cv2
from ultralytics import YOLO

# Khởi tạo camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load model YOLO
model = YOLO("/home/pi/Documents/best_stop.pt")

try:
    while True:
        frame = picam2.capture_array()  # Lấy ảnh numpy array, shape (480, 640, 3), dtype=uint8 RGB

        # Chạy model YOLO trên frame
        results = model(frame)

        # Vẽ kết quả lên frame (model có method .plot())
        annotated_frame = results[0].plot()

        # Hiển thị ảnh kết quả
        cv2.imshow("Camera CSI - YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.close()
    cv2.destroyAllWindows()