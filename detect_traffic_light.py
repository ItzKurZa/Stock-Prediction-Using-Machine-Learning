from ultralytics import YOLO
import cv2

# Tải mô hình YOLO đã train
model = YOLO("best.pt")  # Đường dẫn tới mô hình của bạn

# Đọc ảnh test (bạn có thể thay bằng ảnh/video tùy ý)
cap = cv2.VideoCapture(0)  # webcam: 0; hoặc thay bằng đường dẫn video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25)
    for result in results:
        for box in result.boxes:
        
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            color = (0, 0, 255) if label == 'red' else \
                    (0, 255, 255) if label == 'yellow' else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Traffic Light Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
