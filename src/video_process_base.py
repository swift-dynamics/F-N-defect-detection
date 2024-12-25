import os
from multiprocessing import Queue
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Union
import cv2
import numpy as np
import dotenv
from .setting_mode import ROICoordinates

dotenv.load_dotenv(dotenv_path='./.env',override=True)
dotenv.load_dotenv(dotenv_path='./setting.env', override=True)

# Load environment variables
ROI_X = int(os.getenv('X', 0))
ROI_Y = int(os.getenv('Y', 0))
ROI_W = int(os.getenv('WIDTH', 0))
ROI_H = int(os.getenv('HEIGHT', 0))
ALERT_DEBOUNCE_SECONDS = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))

logger = logging.getLogger(__name__)

class AlertProcessBase:
    def __init__(self):
        logger.info("Alert process base initialized.")
        logger.info(f"Alert debounce seconds: {ALERT_DEBOUNCE_SECONDS}")
    
    def setup_alert(self, alert_directory, alert_info) -> None:
        if alert_directory:
            self.alert_info = alert_info
            self.alerted = False
            self.last_alert_time = datetime.min
            self.alert_debounce = ALERT_DEBOUNCE_SECONDS
            
            self.root_dir = os.path.join(os.getcwd(), alert_directory)
            os.makedirs(self.root_dir, exist_ok=True)
        else:
            logger.error("Alert directory is not provided.")
            raise ValueError("Alert directory is not provided.")

    def trigger_alert(self, frame):
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
            defect_image_path = os.path.join(self.root_dir, f"{timestamp}_{self.alert_info}.jpg")
            cv2.imwrite(defect_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Text image saved to: {defect_image_path}")
        else:
            logger.debug("Alert suppressed due to debounce logic.")
            
class VideoProcessBase(AlertProcessBase):
    def __init__(
        self, 
        source: Union[str, int, Queue],
        fps: int = 30, 
        main_window = None, # If None, will not show live view
        process_window = None
    ):
        super().__init__()
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.exposure = 0  # Initial exposure value (adjust based on camera specifications)
        self.roi: Optional[ROICoordinates] = ROICoordinates(ROI_X, ROI_Y, ROI_W, ROI_H)
        self.setup_video_capture(source)
        self.setup_gui(main_window, process_window)
        
    def setup_gui(self, main_window, process_window):
        self.main_window = main_window
        self.process_window = process_window
        
        if self.main_window:
            cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        if self.process_window:
            cv2.namedWindow(self.process_window, cv2.WINDOW_NORMAL)
            
    def setup_video_capture(self, source) -> None:
        logger.debug(f"Setting up camera with source: {source}")
        if isinstance(source, str) or isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                logger.error("Cannot open camera source: %s", source)
                raise ValueError(f"Cannot open camera source: {source}")
            # self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)  # Set initial exposure
        else:
            self.cap: Queue = source
            logger.info("Using provided frame queue for camera source")
    
    def _adjust_exposure(self, change: int):
        self.exposure += change
        # Clamp exposure to a valid range (-13 to 0 for most cameras)
        self.exposure = max(-13, min(0, self.exposure))
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            logging.info("Exposure adjusted to: %d", self.exposure)
    
    def live_view(self, frame, window_name, draw_roi: bool = True):
        """Visualize the extracted text on the frame."""
        if window_name:
            if draw_roi:
                cv2.rectangle(frame, (self.roi.x, self.roi.y), (self.roi.x + self.roi.width, self.roi.y + self.roi.height), (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User exited detection loop via 'q' key")
                return False
            # Control Exposure
            elif key == ord('w'):
                self.adjust_exposure(1)  # Increase exposure
            elif key == ord('s'):
                self.adjust_exposure(-1)  # Decrease exposure
            return True
        else:
            return False
    
    def get_setting_ROI(self, frame: np.array):
        return frame[
            self.roi.y:self.roi.y + self.roi.height,
            self.roi.x:self.roi.x + self.roi.width
        ]
        
    def to_binary(self, image, otsu: bool = False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if otsu:
            return cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)[1] 
    
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
    
    def control_frame_rate(self, start_time):
        elapsed_time = time.time() - start_time
        delay = max(0.025, self.frame_delay - elapsed_time)
        time.sleep(delay)
        logger.debug(f"Frame processing time: {elapsed_time:.3f}s, delay: {delay:.3f}s. fps: {1/delay:.2f}")