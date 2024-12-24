import cv2
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import os
import dotenv
import torch
from typing import Optional, Union
from multiprocessing import Queue
import easyocr
from threading import Thread
from setting_mode import ROICoordinates

# dotenv.load_dotenv(dotenv_path='./.env',override=True)
dotenv.load_dotenv(dotenv_path='./setting.env', override=True)

# Load environment variables
ROI_X = int(os.getenv('X', 0))
ROI_Y = int(os.getenv('Y', 0))
ROI_W = int(os.getenv('WIDTH', 0))
ROI_H = int(os.getenv('HEIGHT', 0))
TEMPLATE_IMAGE = os.getenv('TEMPLATE_PATH', None)

logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self, source: Union[str, int, Queue], fps: int = 30, show_main: bool = True, show_process: bool = True):
        self.alert_directory = os.getenv('ALERT_FILE_NAME', 'text_defected_images')
        self.roi: Optional[ROICoordinates] = ROICoordinates(ROI_X, ROI_Y, ROI_W, ROI_H)
        self.show_main = show_main
        self.show_process = show_process
        self.setup_camera(source, fps, show_main, show_process)
        self.setup_alert()
        self.setup_ocr()

    def setup_alert(self,) -> None:
        self.alert_info = 'text-defectd'
        self.alerted = False
        self.last_alert_time = datetime.min
        self.alert_debounce = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))
        self.root_dir = os.path.join(os.getcwd(), self.alert_directory)

    def setup_camera(self, camera_source, fps, show_main, show_process) -> None:
        """Initialize and validate camera connection."""
        logger.debug(f"Setting up camera with source: {camera_source}")

        self.exposure = 0  # Initial exposure value (adjust based on camera specifications)
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.window_name = "OCR-main-display"
        self.ocr_window_name = "OCR-process-display"

        if show_main:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if show_process:
            cv2.namedWindow(self.ocr_window_name, cv2.WINDOW_NORMAL)

        if isinstance(camera_source, str) or isinstance(camera_source, int):
            self.cap = cv2.VideoCapture(camera_source)
            if not self.cap.isOpened():
                logger.error("Cannot open camera source: %s", camera_source)
                raise ValueError(f"Cannot open camera source: {camera_source}")
            
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)  # Set initial exposure
        else:
            self.cap: Queue = camera_source
            logger.info("Using provided frame queue for camera source")

    def adjust_exposure(self, change: int):
        """
        Adjust the camera exposure dynamically.

        Args:
            change (int): Amount to adjust the exposure by (positive or negative).
        """
        self.exposure += change
        # Clamp exposure to a valid range (-13 to 0 for most cameras)
        self.exposure = max(-13, min(0, self.exposure))
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            logging.info("Exposure adjusted to: %d", self.exposure)

    def visualize(self, frame, text, window_name):
        """
        Visualize the extracted text on the frame.
        """
        if text:
            cv2.putText(frame, str(text), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # cv2.putText(frame, "Text:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            logger.info("User exited detection loop via 'q' key")
            return False
        elif key == ord('w'):  # Arrow Up Key
            self.adjust_exposure(1)  # Increase exposure
        elif key == ord('s'):  # Arrow Down Key
            self.adjust_exposure(-1)  # Decrease exposure

        return True
    
    def setup_ocr(self) -> None:
        if not torch.cuda.is_available():
            logger.warning("GPU is not available. Falling back to CPU for EasyOCR.")
            self.reader = easyocr.Reader(['en', 'th'], gpu=False)
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} for EasyOCR.")
            self.reader = easyocr.Reader(['en', 'th'], gpu=True)

    def _text_extractor(self, image):
        try:
            results = self.reader.readtext(image, detail=0)  # Extract text, disable detailed output
            output_text = [line.strip() for line in results if line.strip()]
            logger.debug("Extracted text: %s", output_text)
            return output_text if output_text else None
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return None

    def create_roi(self, image: np.array, roi: ROICoordinates):
        return image[
            roi.y:self.roi.y + roi.height,
            roi.x:self.roi.x + roi.width
        ]
        
    def to_binary(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    def trigger_alert(self, frame, alert_info: str):
        """
        Process the extracted text and take appropriate action.
        """
        current_time = datetime.now()
        if not self.alerted or (current_time - self.last_alert_time > timedelta(seconds=self.alert_debounce)):
            # Trigger alert
            self.alerted = True
            self.last_alert_time = current_time

            # Save defected image
            timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            defect_image_path = os.path.join(self.root_dir, f"{timestamp}_{alert_info}.jpg")
            os.makedirs(self.root_dir, exist_ok=True)
            cv2.imwrite(defect_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Text image saved to: {defect_image_path}")
        else:
            logger.debug("Alert suppressed due to debounce logic.")
    
    def _alert_process(self, frame, text, no_of_text=3):
        if text and len(text) > no_of_text:
            self.trigger_alert(frame, self.alert_info)

    def control_frame_rate(self, frame_delay, start_time):
        elapsed_time = time.time() - start_time
        delay = max(0.025, frame_delay - elapsed_time)
        time.sleep(delay)
        logger.debug(f"Frame processing time: {elapsed_time:.3f}s, delay: {delay:.3f}s. fps: {1/delay:.2f}")

    def get_frame(self):
        """
        Retrieve a frame either from the Queue or directly from the webcam.
        """
        if type(self.cap) is cv2.VideoCapture:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                return None
        else:
            if not self.cap.empty():
                frame = self.cap.get()
                if frame is None:
                    logger.error("Failed to read frame from queue")
                    return None
            else:
                logger.warning("Frame queue is empty.")
                return None
        return frame
        
    def cleanup(self):
        if self.cap and type(self.cap) is cv2.VideoCapture:
            self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        try:
            while True:
                start_time = time.time()
                
                frame = self.get_frame()
                if frame is None:
                    logger.info("No frame available. Skipping iteration.")
                    time.sleep(0.1)  # Avoid busy-waiting
                    continue

                processed_image = self.create_roi(frame, self.roi)
                processed_image = self.to_binary(processed_image)
                output_text = self._text_extractor(processed_image)
                logger.debug("Extracted text: %s", output_text)
                
                self.t = Thread(target=self._alert_process, args=(frame, self.alert_info, 3))
                self.t.start()

                if self.show_main:
                    cv2.rectangle(frame, (self.roi.x, self.roi.y), (self.roi.x + self.roi.width, self.roi.y + self.roi.height), (0, 255, 0), 2)
                    if not self.visualize(frame, text=None, window_name=self.window_name):
                        break
                if self.show_process:
                    if not self.visualize(processed_image, text=None, window_name=self.ocr_window_name):
                        break

                # Control frame rate
                self.control_frame_rate(self.frame_delay, start_time)
        
        except KeyboardInterrupt:
            logger.info("Shutting down.")

        finally:
            self.cleanup()

if __name__ == "__main__":
    CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', 0)
    FPS = int(os.getenv('FPS', 30))

    text_extractor = TextExtractor(source=CAMERA_SOURCE, fps=FPS)
    text_extractor.run()