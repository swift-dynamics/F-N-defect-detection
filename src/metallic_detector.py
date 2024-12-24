import cv2
import numpy as np
import dotenv
import os
import time
from datetime import datetime, timedelta
from multiprocessing import Queue
from datetime import datetime
import logging
from threading import Thread
from typing import Tuple, Optional, Union
from .setting_mode import ROICoordinates

logger = logging.getLogger(__name__)

dotenv.load_dotenv(dotenv_path='./setting.env', override=True)

# Load environment variables
ROI_X = int(os.getenv('X', 0))
ROI_Y = int(os.getenv('Y', 0))
ROI_W = int(os.getenv('WIDTH', 0))
ROI_H = int(os.getenv('HEIGHT', 0))
TEMPLATE_IMAGE = os.getenv('TEMPLATE_PATH', None)

class MetallicDetector:
    def __init__(self, camera_source: Union[str, int, Queue], threshold: Union[int, float] = 0.5, fps=30, show_main: bool = True, show_process: bool = True) -> None:
        """Initialize the MetallicDetector with camera and ROI parameters."""
        logger.debug(f"Initializing MetallicDetector with camera_source={camera_source}, threshold={threshold}")
        self.fps = fps
        self.exposure = 0
        self._setup_camera(camera_source)
        self._setup_display_parameters(fps, show_main, show_process)
        self._init_template_image(TEMPLATE_IMAGE)
        self.roi: Optional[ROICoordinates] = ROICoordinates(ROI_X, ROI_Y, ROI_W, ROI_H)
        
        # Alert parameters
        self.threshold = threshold # Threshold for similarity
        self.alerted = False
        self.last_alert_time = datetime.min
        self.alert_debounce = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))
        # Make directory to save defected images
        os.makedirs('defected_images', exist_ok=True)
        self.root_dir = os.path.join(os.getcwd(), 'milk_carton_defected_images')

        logger.info("MetallicDetector initialized successfully")

    def _setup_display_parameters(self, fps: int, show_main, show_process) -> None:
        """Setup display window and frame rate parameters."""
        logger.debug(f"Setting up display parameters with fps: {fps}")
        self.frame_delay = 1.0 / fps
        self.main_disp = 'Metallic-main-display'
        self.sub_disp = 'Metallic-process-display'
        self.show_main = show_main
        self.show_process = show_process

        if self.show_main:
            cv2.namedWindow(self.main_disp, cv2.WINDOW_NORMAL)
        if self.show_process:
            cv2.namedWindow(self.sub_disp, cv2.WINDOW_NORMAL)

    def _setup_camera(self, camera_source) -> None:
        """Initialize and validate camera connection."""
        logger.debug(f"Setting up camera with source: {camera_source}")
        
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

    def _init_template_image(self, template_image: str) -> None:
        """Initialize the template image for comparison."""
        if template_image:
            logger.debug(f"Loading template image from: {template_image}")
            self.template_image = cv2.imread(template_image)
            if self.template_image is not None:
                self.hist_template_image = self._histogramize(self.template_image)
                logger.info("Template image loaded and histogramized")
            else:
                logger.error("Failed to load template image")
                self.template_image = None
        else:
            logger.info("No template image provided")
            self.template_image = None

    def _draw_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Draw ROI rectangle on frame if ROI is set."""
        if self.roi:
            logger.debug(f"Drawing ROI: {self.roi}")
            roi_frame = frame[
                self.roi.y:self.roi.y + self.roi.height,
                self.roi.x:self.roi.x + self.roi.width
            ]
            cv2.rectangle(
                frame,
                (self.roi.x, self.roi.y),
                (self.roi.x + self.roi.width, self.roi.y + self.roi.height),
                (0, 255, 0),
                2
            )
            return frame, roi_frame
        return frame, None

    def _histogramize(self, roi_frame: np.ndarray) -> np.ndarray:
        """Generate normalized HSV histogram for the input image."""
        hsv_image = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
        hist_image = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_image /= hist_image.sum()  # Normalize the histogram
        logger.debug("Generated histogram for ROI frame")
        return hist_image

    def detect_metallic(self, roi_frame: np.ndarray) -> float:
        """Detect metallic objects in the ROI frame."""
        roi_hist = self._histogramize(roi_frame)
        similarity = cv2.compareHist(self.hist_template_image, roi_hist, cv2.HISTCMP_CORREL)
        logger.debug(f"Calculated similarity: {similarity:.2f}")
        return similarity

    def _alert_process(self, frame: np.ndarray, similarity: float, info="metallic_defected") -> None:
        """Process the extracted text and take appropriate action."""
        current_time = datetime.now()
        if similarity > self.threshold:
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

    def get_frame(self):
        """
        Retrieve a frame either from the Queue or directly from the webcam.
        """
        if type(self.cap) is cv2.VideoCapture:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                return None
            return frame
        else:
            if not self.cap.empty():
                frame = self.cap.get()
                logger.debug("Retrieved frame from queue.")
                return frame
            else:
                logger.warning("Frame queue is empty.")
                time.sleep(0.1)
                return np.zeros_like(np.array([0, 0, 0]))  # Return a dummy frame to keep the process running
            
    def run(self) -> None:
        """Process frames to detect metallic objects."""
        logger.info("Starting metallic detection")
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

            frame, roi_frame = self._draw_roi(frame)
            if roi_frame is not None:
                similarity = self.detect_metallic(roi_frame)
                cv2.putText(frame, f"Defected: {similarity:.2f}", (self.roi.x-self.roi.width//2, self.roi.y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                self.t = Thread(target=self._alert_process, args=(frame, similarity))
                # self.t.daemon = True
                self.t.start()
                # self._alert_process(roi_frame, similarity)
                if self.show_process:
                    cv2.imshow(self.sub_disp, roi_frame)

            if self.show_main:
                cv2.imshow(self.main_disp, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User exited detection loop via 'q' key")
                break
            elif key == ord('w'):  # Arrow Up Key
                self.adjust_exposure(1)  # Increase exposure
            elif key == ord('s'):  # Arrow Down Key
                self.adjust_exposure(-1)  # Decrease exposure

            # Control frame rate
            elapsed_time = time.time() - start_time
            delay = max(0.025, self.frame_delay - elapsed_time)
            time.sleep(delay)
            logger.debug(f"Frame processing time: {elapsed_time:.3f}s, delay: {delay:.3f}s. fps: {1/delay:.2f}")
        
    def cleanup(self) -> None:
        """Release resources and close windows."""
        logger.info("Cleaning up resources")
        self.t.join()
        if type(self.cap) is cv2.VideoCapture:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":    
    try:
        detector = MetallicDetector(camera_source="data/Relaxing_highway_traffic.mp4", threshold=0.75, fps=30)
        detector.run()
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
    finally:
        detector.cleanup()