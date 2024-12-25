import numpy as np
import time
import logging
import os
import torch
from typing import Union
from multiprocessing import Queue
import easyocr
from threading import Thread
from .video_process_base import VideoProcessBase

logger = logging.getLogger(__name__)

class TextExtractor(VideoProcessBase):
    def __init__(
        self,
        source: Union[str, int, Queue],
        fps: int = 30, 
        main_window: str = "OCR-main-display", 
        process_window: str = "OCR-process-display",
        alert_info: str = "text-defected",
        alert_directory: str = None,
        text_threshold: int = 3
    ):
        super().__init__(source, fps, main_window, process_window)
        self.alert_directory = alert_directory
        self.alert_info = alert_info
        self.text_threshold = text_threshold
        self.setup_alert(self.alert_directory, self.alert_info)
        self.setup_ocr()

        logger.info("------------------------ TextExtractor initialized ------------------------")
        logger.info(f"Alert Directory: {self.alert_directory}")
        logger.info(f"Alert Info: {self.alert_info}")
        logger.info(f"Text Threshold: {self.text_threshold}")
        logger.info("---------------------------------------------------------------------------") 

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
        
    def _alert_process(self, frame: np.array, text: list, text_threshold: int):
        if text and len(text) > text_threshold:
            detected_text = ''.join(text)
            logger.debug(f"Text detected: {detected_text}")
            self.trigger_alert(frame)

    def run(self):
        try:
            while True:
                start_time = time.time()
                
                frame = self.get_frame()
                if frame is None:
                    # logger.info("No frame available. Skipping iteration.")
                    time.sleep(0.1)  # Avoid busy-waiting
                    continue

                processed_frame = self.get_setting_ROI(frame)
                processed_frame = self.to_binary(processed_frame)
                output_text = self._text_extractor(processed_frame)
                logger.debug("Extracted text: %s", output_text)
                
                self.t = Thread(target=self._alert_process, args=(frame, output_text, self.text_threshold))
                self.t.start()

                if self.main_window and not self.live_view(frame, window_name=self.main_window, color=(255,0,255)):
                    break
                if self.process_window and not self.live_view(processed_frame, window_name=self.process_window, color=(255,0,255)):
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
    ALERT_DIRECTORY = str(os.getenv('TEXT_DEFECTED_ALERT_DIRECTORY', "data/alerts"))
    TEXT_THRSHOLD = int(os.getenv('TEXT_THRSHOLD', 3))

    text_extractor = TextExtractor(
        source=CAMERA_SOURCE, 
        fps=FPS,
        alert_directory=ALERT_DIRECTORY,
        text_threshold=TEXT_THRSHOLD,
        main_window="OCR-main-display",
        process_window="OCR-process-display",
        alert_info="text-defected",
    )
    text_extractor.run()