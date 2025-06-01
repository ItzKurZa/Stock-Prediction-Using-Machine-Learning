from PIL import Image
import numpy as np

class Picamera2:
    def __init__(self):
        self.camera_config = None
        self.is_running = False
        self._test_image_path = "test.jpg"
    
    def create_still_configuration(self, **kwargs):
        self.camera_config = kwargs
        return self.camera_config
    
    def create_preview_configuration(self, **kwargs):
        self.camera_config = kwargs
        return self.camera_config
        
    def configure(self, config):
        self.camera_config = config
    
    def start(self):
        self.is_running = True
        
    def stop(self):
        self.is_running = False
    
    def capture_array(self):
        # Load and return the test image as a numpy array
        img = Image.open(self._test_image_path)
        return np.array(img)
    
    def capture_file(self, filename):
        # Copy the test image to the specified filename
        img = Image.open(self._test_image_path)
        img.save(filename)
        
    def close(self):
        self.stop() 