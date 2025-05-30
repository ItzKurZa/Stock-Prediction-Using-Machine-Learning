import cv2
import os
import time
from ultralytics import YOLO
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Định nghĩa đường dẫn cơ sở
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_2")

# Cấu hình hiển thị
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR_GREEN = (0, 255, 0)
TEXT_COLOR_RED = (0, 0, 255)
TEXT_COLOR_YELLOW = (0, 255, 255)
TEXT_COLOR_WHITE = (255, 255, 255)

# Cấu hình GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor:
    def __init__(self, ENA, IN1, IN2, ENB, IN3, IN4):
        """Khởi tạo điều khiển 2 động cơ
        ENA, IN1, IN2: điều khiển động cơ bên trái
        ENB, IN3, IN4: điều khiển động cơ bên phải
        """
        self.ENA = ENA
        self.IN1 = IN1
        self.IN2 = IN2
        self.ENB = ENB
        self.IN3 = IN3
        self.IN4 = IN4
        
        # Thiết lập các chân GPIO
        GPIO.setup(self.ENA, GPIO.OUT)
        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        GPIO.setup(self.ENB, GPIO.OUT)
        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)
        
        # Khởi tạo PWM cho cả hai động cơ
        self.pwmA = GPIO.PWM(self.ENA, 100)  # Tần số 100Hz
        self.pwmB = GPIO.PWM(self.ENB, 100)
        self.pwmA.start(0)
        self.pwmB.start(0)
    
    def move_forward(self, speed=70):
        """Di chuyển tiến với tốc độ được chỉ định"""
        self.pwmA.ChangeDutyCycle(speed)
        self.pwmB.ChangeDutyCycle(speed)
        
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
    
    def move_slow(self, speed=30):
        """Di chuyển chậm với tốc độ được chỉ định"""
        self.pwmA.ChangeDutyCycle(speed)
        self.pwmB.ChangeDutyCycle(speed)
        
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
    
    def stop(self):
        """Dừng cả hai động cơ"""
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)
        
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
    
    def cleanup(self):
        """Dọn dẹp GPIO"""
        self.pwmA.stop()
        self.pwmB.stop()
        GPIO.cleanup()

def load_model():
    """Tải mô hình YOLOv8 đã được training"""
    try:
        # Tải mô hình YOLOv8 đã được training
        model_path = os.path.join(MODELS_DIR, "yolov8_traffic_light", "weights", "best.pt")
        model = YOLO(model_path)
        print("Đã tải xong mô hình YOLOv8")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

def draw_fps(frame, fps):
    """Hiển thị FPS lên frame"""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), FONT, FONT_SCALE, TEXT_COLOR_WHITE, FONT_THICKNESS)
    return frame

def draw_prediction(frame, light_signal, confidence):
    """Hiển thị kết quả dự đoán lên frame"""
    if light_signal is None:
        text = f"Không phát hiện đèn ({confidence:.2f})"
        color = TEXT_COLOR_WHITE
    else:
        text = f"Đèn: {light_signal} ({confidence:.2f})"
        if light_signal == 'red':
            color = TEXT_COLOR_RED
        elif light_signal == 'green':
            color = TEXT_COLOR_GREEN
        else:
            color = TEXT_COLOR_YELLOW
    
    cv2.putText(frame, text, (10, 70), FONT, FONT_SCALE, color, FONT_THICKNESS)
    return frame

def draw_status(frame, status):
    """Hiển thị trạng thái xe lên frame"""
    cv2.putText(frame, f"Trạng thái: {status}", (10, frame.shape[0] - 20), 
                FONT, FONT_SCALE, TEXT_COLOR_WHITE, FONT_THICKNESS)
    return frame

def detect_traffic_light(frame, model):
    """Phát hiện và phân loại đèn giao thông bằng YOLOv8"""
    try:
        # Cải thiện độ tương phản và độ sáng
        frame_processed = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        # Phát hiện đối tượng bằng YOLOv8
        results = model.predict(frame_processed, conf=0.3)
        
        if not results[0].boxes:
            return frame, None, 0.0
        
        # Lấy kết quả dự đoán tốt nhất
        best_box = results[0].boxes[0]
        cls = int(best_box.cls.cpu().numpy()[0])
        conf = float(best_box.conf.cpu().numpy()[0])
        
        # Chuyển đổi class ID thành tên
        class_names = {0: 'green', 1: 'red', 2: 'yellow'}
        light_signal = class_names.get(cls)
        
        if light_signal is None:
            return frame, None, 0.0
        
        # Vẽ kết quả
        x1, y1, x2, y2 = map(int, best_box.xyxy.cpu().numpy()[0])
        color = (0, 0, 255) if light_signal == 'red' else (0, 255, 0) if light_signal == 'green' else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{light_signal}: {conf*100:.1f}%"
        cv2.putText(frame, label_text, (x1, y1-10), FONT, FONT_SCALE, color, FONT_THICKNESS)
        
        return frame, light_signal, conf
        
    except Exception as e:
        print(f"Lỗi khi phát hiện đèn giao thông: {e}")
        return frame, None, 0.0

def control_vehicle(motor, light_signal, confidence):
    """Điều khiển xe dựa trên tín hiệu đèn giao thông"""
    if light_signal is None or confidence < 0.7:
        motor.move_slow()
        return "Di chuyển chậm"
    
    if light_signal == 'red':
        motor.stop()
        return "Dừng lại"
    elif light_signal == 'yellow':
        motor.move_slow()
        return "Di chuyển chậm"
    else:  # green
        motor.move_forward()
        return "Di chuyển bình thường"

def main():
    # Khởi tạo camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (DISPLAY_WIDTH, DISPLAY_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    
    # Khởi tạo motor
    motor = Motor(ENA=17, IN1=27, IN2=22, ENB=23, IN3=24, IN4=25)
    
    # Tải model
    model = load_model()
    if model is None:
        print("Không thể tải model. Kết thúc chương trình.")
        return
    
    # Biến để tính FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Đọc frame từ camera
            frame = picam2.capture_array()
            
            # Phát hiện đèn giao thông
            frame, light_signal, confidence = detect_traffic_light(frame, model)
            
            # Điều khiển xe
            status = control_vehicle(motor, light_signal, confidence)
            
            # Tính FPS
            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Vẽ thông tin lên frame
            frame = draw_fps(frame, fps)
            frame = draw_prediction(frame, light_signal, confidence)
            frame = draw_status(frame, status)
            
            # Hiển thị frame
            cv2.imshow('Traffic Light Detection', frame)
            
            # Thoát nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Chương trình bị dừng bởi người dùng")
    finally:
        # Dọn dẹp
        motor.cleanup()
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()