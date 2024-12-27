import cv2
import numpy as np
import os
import time
from multiprocessing import Queue
import logging
from threading import Thread
from typing import Union
from utils import VideoProcessBase, AlertProcessBase

logger = logging.getLogger(__name__)

class HistogramComparator(VideoProcessBase, AlertProcessBase):
    def __init__(self, source: Union[str, int, Queue], fps: int = 30, 
                 main_window: str = None, process_window: str = None,
                 alert_info: str = "text-defected", alert_directory: str = None,
                 simm_threshold: float = 0.5, template_image: str = None, 
                 save: bool = False, minio: bool = False):
        VideoProcessBase.__init__(self, source, fps, main_window, process_window)
        AlertProcessBase.__init__(self, save, minio)
        self.alert_directory = alert_directory
        self.alert_info = alert_info
        self.simm_threshold = simm_threshold
        self.setup_alert(alert_directory, alert_info)
        self.hist_template_image = self._load_and_histogramize(template_image)

        logger.info("------------------------ HistogramComparator initialized ------------------------")
        logger.info(f"Alerts will be saved to: {'local' if save else 'MinIO' if minio else 'No storage'}")
        logger.info(f"Similarity Threshold: {simm_threshold}")
        logger.info("------------------------------------------------------------------------------")

    def _load_and_histogramize(self, template_path: str) -> np.ndarray:
        if not template_path:
            logger.error("No template image provided.")
            raise ValueError("No template image provided.")
        template_image = cv2.imread(template_path)
        if template_image is None:
            raise ValueError(f"Could not load template image from {template_path}")
        return self._histogramize(template_image)

    def _histogramize(self, image: np.ndarray) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
        hist_image = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return hist_image / hist_image.sum()

    def _compare_hist_template(self, hist_data: np.ndarray) -> float:
        return cv2.compareHist(hist_data, self.hist_template_image, cv2.HISTCMP_CORREL)

    def _alert_process(self, frame: np.ndarray, similarity: float):
        if similarity >= self.simm_threshold:
            logger.debug(f"Metallic detected: {similarity:.3f}")
            self.trigger_alert(frame, info=f"Metallic defected (%): {similarity*100:.3f}")

    # (Optinal) Future work!
    # def run_multi_camera(self):
    #     """Run all camera sources in parallel."""
    #     try:
    #         threads = []
    #         for source in self.sources:
    #             thread = Thread(target=self.run_camera_source, args=(source,))
    #             thread.start()
    #             threads.append(thread)

    #         # Join threads to ensure they all finish
    #         for thread in threads:
    #             thread.join()

    #     except KeyboardInterrupt:
    #         logger.info("Shutting down all camera sources.")

    #     finally:
    #         logger.info("Cleanup completed.")
    #         self.cleanup()
            
    def run(self):
        try:
            while True:
                start_time = time.time()
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                roi_frame = self.get_setting_ROI(frame.copy())
                hist_roi_frame = self._histogramize(roi_frame)
                similarity = self._compare_hist_template(hist_roi_frame)

                Thread(target=self._alert_process, args=(roi_frame, similarity)).start()

                frame = cv2.putText(frame, f"Metallic Defected (%): {similarity*100:.3f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if self.main_window and not self.live_view(frame=frame, window_name=self.main_window, color=(0, 255, 255), 
                                                           draw_roi=True, text=f"No. of alert: {self.alert_count}"):
                    break
                if self.process_window and not self.live_view(frame=roi_frame, window_name=self.process_window, color=(0, 255, 255), 
                                                              draw_roi=False, text=None):
                    break

                self.control_frame_rate(start_time)
        
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            self.cleanup()

if __name__ == "__main__":
    CAMERA_SOURCE = os.getenv('CAMERA_SOURCE')
    FPS = int(os.getenv('FPS', 30))
    ALERT_DIRECTORY = os.getenv("METALLIC_DEFECTED_ALERT_DIRECTORY")
    THRESHOLD = float(os.getenv('THRESHOLD', 0.5))
    TEMPLATE_IMAGE = os.getenv('TEMPLATE_IMAGE')

    detector = HistogramComparator(
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