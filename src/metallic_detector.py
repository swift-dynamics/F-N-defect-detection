import cv2
import numpy as np
import dotenv
import os
import time
import multiprocessing
from multiprocessing import Process, Queue
from datetime import datetime
import logging
from threading import Thread
from typing import Tuple, Optional, Union
from .setting_mode import ROICoordinates

dotenv.load_dotenv(dotenv_path='./setting.env', override=True)

# Load environment variables
ROI_X = int(os.getenv('X', 0))
ROI_Y = int(os.getenv('Y', 0))
ROI_W = int(os.getenv('WIDTH', 0))
ROI_H = int(os.getenv('HEIGHT', 0))
TEMPLATE_IMAGE = os.getenv('TEMPLATE_PATH', None)

class MetallicDetector:
    def __init__(self, camera_source: Union[str, int, Queue], threshold: Union[int, float] = 0.5, fps=30) -> None:
        """Initialize the MetallicDetector with camera and ROI parameters."""
        logging.debug(f"Initializing MetallicDetector with camera_source={camera_source}, threshold={threshold}")
        self._setup_camera(camera_source)
        self._setup_display_parameters(fps)
        self._init_template_image(TEMPLATE_IMAGE)
        self.roi: Optional[ROICoordinates] = ROICoordinates(ROI_X, ROI_Y, ROI_W, ROI_H)
        # Alert parameters
        self.alerted = False
        # To reduce false positives
        self.no_same_alerts = 0
        self.alert_debounce = 10
        self.threshold = threshold # Threshold for similarity
        self.no_of_defects = 0 # Number of defects detected
        # Make directory to save defected images
        os.makedirs('defected_images', exist_ok=True)
        self.root_dir = os.path.join(os.getcwd(), 'defected_images')
        logging.info("MetallicDetector initialized successfully")

    def _setup_display_parameters(self, fps: int) -> None:
        """Setup display window and frame rate parameters."""
        logging.debug(f"Setting up display parameters with fps: {fps}")
        self.frame_delay = 1.0 / fps
        self.main_disp = 'Metallic Detector'
        self.sub_disp = 'ROI Frame'
        cv2.namedWindow(self.main_disp, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.sub_disp, cv2.WINDOW_NORMAL)

    def _setup_camera(self, camera_source) -> None:
        """Initialize and validate camera connection."""
        logging.debug(f"Setting up camera with source: {camera_source}")
        if isinstance(camera_source, str) or isinstance(camera_source, int):
            self.cap = cv2.VideoCapture(camera_source)
            if not self.cap.isOpened():
                logging.error("Failed to access camera/video source")
                raise ValueError("Failed to access camera/video source")
        else:
            self.cap: Queue = camera_source
            logging.info("Using provided frame queue for camera source")

    def _init_template_image(self, template_image: str) -> None:
        """Initialize the template image for comparison."""
        if template_image:
            logging.debug(f"Loading template image from: {template_image}")
            self.template_image = cv2.imread(template_image)
            if self.template_image is not None:
                self.hist_template_image = self._histogramize(self.template_image)
                logging.info("Template image loaded and histogramized")
            else:
                logging.error("Failed to load template image")
                self.template_image = None
        else:
            logging.info("No template image provided")
            self.template_image = None

    def _draw_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Draw ROI rectangle on frame if ROI is set."""
        if self.roi:
            logging.debug(f"Drawing ROI: {self.roi}")
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
        logging.debug("Generated histogram for ROI frame")
        return hist_image

    def detect_metallic(self, roi_frame: np.ndarray) -> float:
        """Detect metallic objects in the ROI frame."""
        roi_hist = self._histogramize(roi_frame)
        similarity = cv2.compareHist(self.hist_template_image, roi_hist, cv2.HISTCMP_CORREL)
        logging.debug(f"Calculated similarity: {similarity:.2f}")
        return similarity

    def _alert_process(self, frame: np.ndarray, similarity: float, info="metallic_defected") -> None:
        """Alert the user if metallic object is detected."""
        if similarity > self.threshold:
            self.no_same_alerts += 1
            if self.no_same_alerts > self.alert_debounce and not self.alerted:
                # Reset alert debounce
                self.alerted = True
                self.no_same_alerts = 0
                logging.info(f"Metallic object detected! Similarity: {similarity:.2f}")
                self.no_of_defects += 1
                # Save the defected image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                defect_image_path = f"{self.root_dir}/{timestamp}_{info}.jpg"
                cv2.imwrite(defect_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logging.info(f"Defected image saved to: {defect_image_path}")

    def run(self) -> None:
        """Process frames to detect metallic objects."""
        logging.info("Starting metallic detection")
        while True:
            start_time = time.time()

            if type(self.cap) is multiprocessing.queues.Queue:
                try:
                    frame = self.cap.get()
                    if frame is None:
                        logging.error("Failed to read frame from queue")
                        break
                except:
                    time.sleep(0.1)
                    continue
            else:
                print(type(self.cap))
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to read frame from camera")
                    break

            frame, roi_frame = self._draw_roi(frame)
            if roi_frame is not None:
                similarity = self.detect_metallic(roi_frame)
                cv2.putText(frame, f"Defected: {similarity:.2f}", (self.roi.x-self.roi.width//2, self.roi.y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                self.t = Thread(target=self._alert_process, args=(frame, similarity))
                self.t.daemon = True
                self.t.start()
                # self._alert_process(roi_frame, similarity)
                cv2.imshow(self.sub_disp, roi_frame)

            cv2.imshow(self.main_disp, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User exited detection loop via 'q' key")
                break

            # Control frame rate
            elapsed_time = time.time() - start_time
            delay = max(0.01, self.frame_delay - elapsed_time)
            time.sleep(delay)
            logging.debug(f"Frame processing time: {elapsed_time:.3f}s, delay: {delay:.3f}s. fps: {1/delay:.2f}")
        
    def cleanup(self) -> None:
        """Release resources and close windows."""
        logging.info("Cleaning up resources")
        self.t.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setting Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
            format='%(asctime)s - %(message)s', 
            datefmt='%d-%b-%y %H:%M:%S',
            level=logging.DEBUG if args.debug else logging.INFO
    )
    
    try:
        detector = MetallicDetector(camera_source="data/Relaxing_highway_traffic.mp4", threshold=0.75, fps=30)
        detector.run()
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
    finally:
        detector.cleanup()