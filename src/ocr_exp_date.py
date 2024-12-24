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
from video_process_base import VideoProcessBase

logger = logging.getLogger(__name__)
        
class TextExtractor(VideoProcessBase):
    def __init__(
        self,
        source: Union[str, int, Queue],
        fps: int = 30, 
        main_window: str = None, 
        process_window: str = "process-display",
        root_dir: str = "text-defected-images"
    ):
        super().__init__(source, fps, main_window, process_window)
        self.alert_directory = os.getenv('ALERT_FILE_NAME', 'text_defected_images')
        self.setup_alert()
        self.setup_ocr()

    def setup_alert(self,) -> None:
        self.alert_info = 'text-defected'
        self.alerted = False
        self.last_alert_time = datetime.min
        self.alert_debounce = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))
        
        self.root_dir = os.path.join(os.getcwd(), self.alert_directory)
        os.makedirs(self.root_dir, exist_ok=True)
    
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
            cv2.imwrite(defect_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Text image saved to: {defect_image_path}")
        else:
            logger.debug("Alert suppressed due to debounce logic.")
    
    def _alert_process(self, frame: np.array, text: list, no_of_text: int = 3):
        if text and len(text) > no_of_text:
            detected_text = ''.join(text)
            logger.debug(f"Text detected: {detected_text}")
            self.trigger_alert(frame, self.alert_info)

    def run(self):
        try:
            while True:
                start_time = time.time()
                
                frame = self.get_frame()
                if frame is None:
                    logger.info("No frame available. Skipping iteration.")
                    time.sleep(0.1)  # Avoid busy-waiting
                    continue

                processed_frame = self.get_setting_ROI(frame)
                processed_frame = self.to_binary(processed_frame)
                output_text = self._text_extractor(processed_frame)
                logger.debug("Extracted text: %s", output_text)
                
                self.t = Thread(target=self._alert_process, args=(frame, output_text, 3))
                self.t.start()

                if not self.live_view(frame, window_name=self.main_window):
                    break
                if not self.live_view(processed_frame, window_name=self.process_window):
                    break

                # Control frame rate
                self.control_frame_rate(start_time)
        
        except KeyboardInterrupt:
            logger.info("Shutting down.")

        finally:
            self.cleanup()

if __name__ == "__main__":
    CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', "data/Relaxing_highway_traffic.mp4")
    FPS = int(os.getenv('FPS', 30))

    text_extractor = TextExtractor(
        source=CAMERA_SOURCE, 
        fps=FPS,
        main_window="OCR-main-display",
        process_window="OCR-process-display",
        root_dir="text-defected-images"
    )
    text_extractor.run()