import cv2
import numpy as np
import os
import time
from multiprocessing import Queue
import logging
from threading import Thread
from typing import Union
from .video_process_base import VideoProcessBase

logger = logging.getLogger(__name__)

class MetallicDetector(VideoProcessBase):
    def __init__(
        self,
        source: Union[str, int, Queue],
        fps: int = 30, 
        main_window: str = None, 
        process_window: str = None,
        alert_info: str = "text-defected",
        alert_directory: str = None,
        simm_threshold: float = 0.5,
        template_image: str = None
    ):
        super().__init__(source, fps, main_window, process_window)
        self.alert_directory = alert_directory
        self.alert_info = alert_info
        self.simm_threshold = simm_threshold
        self.setup_alert(self.alert_directory, self.alert_info)
        self._setup_template(template_image)

        logger.info("------------------------ MetallicDetector initialized ------------------------")
        logger.info(f"Alert Directory: {self.alert_directory}")
        logger.info(f"Alert Info: {self.alert_info}")
        logger.info(f"Similarity Threshold: {self.simm_threshold}")
        logger.info("------------------------------------------------------------------------------")

    def _setup_template(self, template_path: str) -> None:
        if template_path:
            template_image = cv2.imread(template_path)
            self.hist_template_image = self._histogramize(template_image)
            logger.debug(f"Template image loaded from {template_path}")
        else:
            logger.error("No template image provided.")
            raise ValueError("No template image provided.")

    def _histogramize(self, image: np.ndarray) -> np.ndarray:
        try:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
            hist_image = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
            hist_image /= hist_image.sum()  # Normalize the histogram
        except Exception as e:
            raise ValueError(f"Error during histogram calculation: {e}")
        
        return hist_image

    def _compare_hist_template(self, hist_data) -> float:
        try:
            similarity = cv2.compareHist(
                hist_data, 
                self.hist_template_image,
                cv2.HISTCMP_CORREL
            )
        except Exception as e:
            raise ValueError(f"Error during metallic detection: {e}")
        logger.debug(f"Calculated similarity: {similarity:.2f}")
        return similarity
            
    def _alert_process(self, frame: np.array, similarity, threhold):
        if similarity and similarity >= threhold:
            logger.debug(f"Metallic detected: {similarity:.3f}")
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

                roi_frame = self.get_setting_ROI(frame.copy())
                hist_roi_frame = self._histogramize(roi_frame)
                similarity = self._compare_hist_template(hist_roi_frame)
                logger.debug(f"Calculated similarity: {similarity:.2f}")
                
                self.t = Thread(target=self._alert_process, args=(roi_frame, similarity, self.simm_threshold))
                self.t.start()
                
                if self.main_window and not self.live_view(frame, window_name=self.main_window, color=(0,255,255), draw_roi=True):
                    break
                if self.process_window and not self.live_view(roi_frame, window_name=self.process_window, color=(0,255,255), draw_roi=False):
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
    ALERT_DIRECTORY = str(os.getenv("METALLIC_DEFECTED_ALERT_DIRECTORY", "metallic_defected_images"))
    THRESHOLD = float(os.getenv('THRESHOLD', 0.5))
    TEMPLATE_IMAGE = str(os.getenv('TEMPLATE_IMAGE', None))

    detector = MetallicDetector(
        source=CAMERA_SOURCE, 
        fps=FPS,
        alert_directory=ALERT_DIRECTORY,
        simm_threshold=THRESHOLD,
        template_image=TEMPLATE_IMAGE,
        main_window="Metallic-main-display",
        process_window="Metallic-process-display",
        alert_info="metallic-defected"
    )
    detector.run()