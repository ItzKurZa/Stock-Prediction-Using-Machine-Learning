from mock_picamera2 import Picamera2
import cv2
import numpy as np

def main():
    # Initialize the camera
    camera = Picamera2()
    
    # Configure and start the camera
    config = camera.create_preview_configuration()
    camera.configure(config)
    camera.start()
    
    try:
        # Capture an image
        print("Capturing image...")
        image = camera.capture_array()
        
        # Save the captured image
        print("Saving captured image as 'captured.jpg'...")
        cv2.imwrite("captured.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Display some information about the captured image
        print(f"Image shape: {image.shape}")
        print(f"Image data type: {image.dtype}")
        
        # You can also capture directly to a file
        print("Saving image directly to 'direct_capture.jpg'...")
        camera.capture_file("direct_capture.jpg")
        
    finally:
        # Always close the camera properly
        camera.close()
        print("Camera closed")

if __name__ == "__main__":
    main() 