import os
from multiprocessing import Queue
from threading import Thread
import time
import logging
from typing import Union
import cv2
import numpy as np
import torch
import easyocr
from utils import VideoProcessBase, AlertProcessBase

logger = logging.getLogger(__name__)

class MFDEXDInspector(VideoProcessBase, AlertProcessBase):
    def __init__(
        self,
        source: Union[str, int, Queue],
        fps: int = 30, 
        main_window: str = "OCR-main-display", 
        process_window: str = "OCR-process-display",
        alert_info: str = "text-defected",
        alert_directory: str = None,
        text_threshold: int = 3,
        save: bool = False,
        minio: bool = False
    ):
        """
        Initialize the MFDEXDInspector class.

        Args:
            source (Union[str, int, Queue]): The video source, which can be a file path, 
                camera index, or a Queue object for streaming video.
            fps (int, optional): Frames per second for the video processing. Defaults to 30.
            main_window (str, optional): Name of the main display window. Defaults to "OCR-main-display".
            process_window (str, optional): Name of the processed display window. Defaults to "OCR-process-display".
            alert_info (str, optional): The type of alert. Defaults to "text-defected".
            alert_directory (str, optional): The directory to save alerts to. Defaults to None.
            text_threshold (int, optional): The threshold for text detection. Defaults to 3.
            save (bool, optional): Whether to save alerts locally. Defaults to False.
            minio (bool, optional): Whether to save alerts to MinIO. Defaults to False.
        """
        VideoProcessBase.__init__(self, source, fps, main_window, process_window)
        AlertProcessBase.__init__(self, save, minio)
        
        self.alert_directory = alert_directory
        self.alert_info = alert_info
        self.text_threshold = text_threshold
        self.setup_alert(alert_directory, alert_info)
        self.setup_ocr()

        logger.info("------------------------ MFDEXDInspector initialized ------------------------")
        logger.info(f"Alerts will be saved to: {'local' if save else 'MinIO' if minio else 'No storage'}")        
        logger.info(f"Text Threshold: {text_threshold}")
        logger.info("---------------------------------------------------------------------------") 

    def setup_ocr(self) -> None:
        """
        Set up the EasyOCR module for text extraction.

        If the GPU is available, use it for EasyOCR. Otherwise, fall back to CPU.
        """
        if not torch.cuda.is_available():
            logger.warning("GPU is not available. Falling back to CPU for EasyOCR.")
            self.reader = easyocr.Reader(['en', 'th'], gpu=False)
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} for EasyOCR.")
            self.reader = easyocr.Reader(['en', 'th'], gpu=True)

    def _text_extractor(self, image):
        """
        Extracts text from the given image using EasyOCR.

        Args:
            image: The image from which to extract text.

        Returns:
            A list of extracted text lines, or None if no text was extracted or an error occurred.
        """
        try:
            results = self.reader.readtext(image, detail=0)  # Extract text, disable detailed output
            output_text = [line.strip() for line in results if line.strip()]
            # logger.debug("Extracted text: %s", output_text)
            return output_text if output_text else None
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return None
        
    def _alert_process(self, frame: np.array, text: list, text_threshold: int):
        """
        Processes the extracted text and takes appropriate action.

        Args:
            frame (np.ndarray): The current frame from the video stream.
            text (list): A list of extracted text lines.
            text_threshold (int): The minimum number of text lines required to trigger an alert.
        """
        if text and len(text) > text_threshold:
            detected_text = ''.join(text)
            logger.debug(f"Text detected: {detected_text}")
            self.trigger_alert(frame, info=f"Defected text: {detected_text}")

    def run(self):
        """
        Continuously processes video frames to extract text and trigger alerts.

        This method runs an infinite loop to capture video frames, process them to
        extract text, and trigger alerts based on the extracted content. The loop
        continues until interrupted, at which point resources are cleaned up.

        Raises:
            KeyboardInterrupt: If the process is interrupted manually.
        """
        try:
            while True:
                start_time = time.time()
                
                frame = self.get_frame()
                if frame is None:
                    # logger.info("No frame available. Skipping iteration.")
                    time.sleep(0.1)  # Avoid busy-waiting
                    continue

                processed_frame = self.get_setting_ROI(frame.copy())
                processed_frame = self.to_binary(processed_frame, otsu=True)
                h, w = processed_frame.shape[:2]
                processed_frame = processed_frame[0:h//2, 0:w//2]  # Crop to ROI
                output_text = self._text_extractor(processed_frame)
                # logger.debug("Extracted text: %s", output_text)
                
                self.t = Thread(target=self._alert_process, args=(frame, output_text, self.text_threshold))
                self.t.start()

                cv2.putText(frame, f"Text: {output_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                if self.main_window and not self.live_view(frame=frame, window_name=self.main_window, color=(255,0,255), 
                                                           draw_roi=True, text=f"No. of alert: {self.alert_count}"):
                    break
                if self.process_window and not self.live_view(frame=processed_frame, window_name=self.process_window, color=(255,0,255), 
                                                              draw_roi=False, text=None):
                    break

                # Control frame rate
                self.control_frame_rate(start_time)
        
        except KeyboardInterrupt:
            logger.info("Shutting down.")

        finally:
            self.cleanup()

if __name__ == "__main__":
    CAMERA_SOURCE = os.getenv('CAMERA_SOURCE')
    FPS = int(os.getenv('FPS', 30))
    ALERT_DIRECTORY = str(os.getenv('TEXT_DEFECTED_ALERT_DIRECTORY'))
    TEXT_THRSHOLD = int(os.getenv('TEXT_THRSHOLD', 3))

    text_extractor = MFDEXDInspector(
        source=CAMERA_SOURCE, 
        fps=FPS,
        alert_directory=ALERT_DIRECTORY,
        text_threshold=TEXT_THRSHOLD,
        main_window="OCR-main-display",
        process_window="OCR-process-display",
        alert_info="text-defected",
    )
    text_extractor.run()
