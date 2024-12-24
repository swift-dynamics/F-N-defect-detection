import cv2
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import os
import dotenv
import torch
from typing import Tuple, Optional, Union
from multiprocessing import Queue
import easyocr
from threading import Thread

dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self, source: Union[str, int, Queue], fps: int = 30, show_main: bool = True, show_process: bool = True):
        self.alerted = False
        self.last_alert_time = datetime.min
        self.alert_debounce = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))
        self.root_dir = os.path.join(os.getcwd(), 'text_defected_images')
        # self.frame_queue = frame_queue
        # self.cap = None
        self._setup_camera(source)
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.show_main = show_main
        self.show_process = show_process
        self.window_name = "OCR-main-display"
        self.ocr_window_name = "OCR-process-display"

        if show_main:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if show_process:
            cv2.namedWindow(self.ocr_window_name, cv2.WINDOW_NORMAL)

        # GPU check for EasyOCR
        if not torch.cuda.is_available():
            logger.warning("GPU is not available. Falling back to CPU for EasyOCR.")
            self.reader = easyocr.Reader(['en', 'th'], gpu=False)
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} for EasyOCR.")
            self.reader = easyocr.Reader(['en', 'th'], gpu=True)

    def _setup_camera(self, camera_source) -> None:
        """Initialize and validate camera connection."""
        logger.debug(f"Setting up camera with source: {camera_source}")
        if isinstance(camera_source, str) or isinstance(camera_source, int):
            self.cap = cv2.VideoCapture(camera_source)
            if not self.cap.isOpened():
                logger.error("Failed to access camera/video source")
                raise ValueError("Failed to access camera/video source")
        else:
            self.cap: Queue = camera_source
            logger.info("Using provided frame queue for camera source")

    def text_extractor(self, image):
        try:
            results = self.reader.readtext(image, detail=0)  # Extract text, disable detailed output
            output_text = [line.strip() for line in results if line.strip()]
            logger.debug("Extracted text: %s", output_text)
            return output_text if output_text else None
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return None

    def pre_process(self, image):
        image = cv2.resize(image, (640, 640), cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        crop_image = binary_image[0:binary_image.shape[0] // 2 - 50, 10:binary_image.shape[1] // 2]
        kernel = np.ones((2, 2), np.uint8)
        cropped_morph_image = cv2.morphologyEx(crop_image, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cropped_morph_image
    
    def alert_process(self, frame, output_text, info="text_defected"):
        """
        Process the extracted text and take appropriate action.
        """
        current_time = datetime.now()
        if output_text:
            if not self.alerted or (current_time - self.last_alert_time > timedelta(seconds=self.alert_debounce)):
                # Trigger alert
                self.alerted = True
                self.last_alert_time = current_time

                # Save defected image
                timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                defect_image_path = os.path.join(self.root_dir, f"{timestamp}_{info}.jpg")
                os.makedirs(self.root_dir, exist_ok=True)
                cv2.imwrite(defect_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"Text image saved to: {defect_image_path}")
            else:
                logger.debug("Alert suppressed due to debounce logic.")
        else:
            logger.debug("No defect detected.")
    
    def visualize(self, frame, output_text, window_name):
        """
        Visualize the extracted text on the frame.
        """
        if output_text:
            cv2.putText(frame, str(output_text), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, "Text:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("User exited detection loop via 'q' key")
            return False

        return True
    
    def run(self):
        try:
            while True:
                start_time = time.time()

                if type(self.cap) is cv2.VideoCapture:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.error("Failed to read frame from camera")
                        break
                else:
                    try:
                        frame = self.cap.get()
                        if frame is None:
                            logger.error("Failed to read frame from queue")
                            break
                    except:
                        time.sleep(0.1)
                        continue

                processed_image = self.pre_process(frame.copy())
                output_text = self.text_extractor(processed_image)

                self.t = Thread(target=self.alert_process, args=(frame, output_text))
                self.t.start()
                # self.alert_process(processed_image, output_text)
                
                if self.show_main:
                    if not self.visualize(frame, output_text, window_name=self.window_name):
                        break
                if self.show_process:
                    if not self.visualize(processed_image, output_text, window_name=self.ocr_window_name):
                        break

                # Control frame rate
                elapsed_time = time.time() - start_time
                delay = max(0.025, self.frame_delay - elapsed_time)
                time.sleep(delay)
                logger.debug(f"Frame processing time: {elapsed_time:.3f}s, delay: {delay:.3f}s. fps: {1/delay:.2f}")
        
        except KeyboardInterrupt:
            logger.info("Shutting down.")

        finally:
            self.cleanup()

    def cleanup(self):
        self.t.join()
        if self.cap and type(self.cap) is cv2.VideoCapture:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', 0)
    FPS = int(os.getenv('FPS', 30))

    text_extractor = TextExtractor(source=CAMERA_SOURCE, fps=FPS)
    text_extractor.run()