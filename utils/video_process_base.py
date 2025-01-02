import os
from multiprocessing import Queue
import json
import time
import logging
from typing import Optional, Union
import cv2
import numpy as np
import dotenv
from data import ROICoordinates

# dotenv.load_dotenv(override=True)

# Load environment variables
with open('data/roi_coordinates.json', 'r') as f:
    data = json.load(f)
    ROI_X = data['roi']['X']
    ROI_Y = data['roi']['Y']
    ROI_W = data['roi']['WIDTH']
    ROI_H = data['roi']['HEIGHT']


# ROI_X = int(os.getenv('X', 0))
# ROI_Y = int(os.getenv('Y', 0))
# ROI_W = int(os.getenv('WIDTH', 0))
# ROI_H = int(os.getenv('HEIGHT', 0))

logger = logging.getLogger(__name__)


class VideoProcessBase:
    def __init__(
        self,
        source: Union[str, int, Queue],
        fps: int = 30,
        main_window: Optional[str] = None,
        process_window: Optional[str] = None
    ) -> None:
        """
        Initialize the VideoProcessBase class.

        Args:
            _frame_id (int): The current frame ID.
            source (Union[str, int, Queue]): The video source, which can be a file path, 
                camera index, or a Queue object for streaming video.
            fps (int, optional): Frames per second for the video processing. Defaults to 30.
            main_window (Optional[str], optional): Name of the main display window. Defaults to None.
            process_window (Optional[str], optional): Name of the processed display window. Defaults to None.
        """
        self._frame_id = 0
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.exposure = 0  # Initial exposure value (adjust based on camera specifications)
        self.roi: Optional[ROICoordinates] = ROICoordinates(ROI_X, ROI_Y, ROI_W, ROI_H)
        self.setup_video_capture(source)
        self.setup_gui(main_window, process_window)

    def setup_gui(self, main_window: Optional[str], process_window: Optional[str]) -> None:
        """
        Set up the main and process display windows.

        Args:
            main_window (Optional[str]): Name of the main display window. Defaults to None.
            process_window (Optional[str]): Name of the processed display window. Defaults to None.
        """
        self.main_window = main_window
        self.process_window = process_window

        if main_window:
            cv2.namedWindow(main_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if process_window:
            cv2.namedWindow(process_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def setup_video_capture(self, source: Union[str, int, Queue]) -> None:
        """
        Set up the video capture source.

        Args:
            source (Union[str, int, Queue]): The video source, which can be a file path, 
                camera index, or a Queue object for streaming video.

        Raises:
            ValueError: If the camera source cannot be opened.
        """
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

    def _adjust_exposure(self, change: int) -> None:
        """
        Adjust the camera's exposure setting.

        Args:
            change (int): The amount by which to adjust the exposure.

        This method updates the exposure value based on the input change,
        ensuring it remains within the valid range of -13 to 0. If a video
        capture object is set, it applies the new exposure setting and logs
        the updated value.
        """
        self.exposure += change
        # Clamp exposure to a valid range (-13 to 0 for most cameras)
        self.exposure = max(-13, min(0, self.exposure))
        if self.cap and type(self.cap) is cv2.VideoCapture:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            logger.info("Exposure adjusted to: %d", self.exposure)
        else:
            logger.warning("No camera source available for exposure adjustment.")


    def live_view(self, frame: np.ndarray, window_name: str, 
                  color: tuple, text: str, draw_roi: bool = True) -> bool:
        """
        Display the frame in a window, with an optional ROI rectangle and text.

        Args:
            frame (np.ndarray): The frame to display.
            window_name (str): The name of the window to display in.
            color (tuple): The color of the rectangle and text.
            text (str): The text to display above the ROI rectangle.
            draw_roi (bool, optional): Whether to draw the ROI rectangle. Defaults to True.

        Returns:
            bool: Whether the user has exited the loop by pressing the 'q' key.
        """
        if window_name:
            if draw_roi:
                cv2.rectangle(frame, (self.roi.x, self.roi.y), (self.roi.x + self.roi.width, self.roi.y + self.roi.height), color, 2)
            if text:
                cv2.putText(frame, text, (self.roi.x, self.roi.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if self.exposure > 0:
                cv2.putText(frame, f"Exposure: {self.exposure}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User exited detection loop via 'q' key")
                return False
            # Control Exposure
            elif key == ord('w'):
                self._adjust_exposure(1)  # Increase exposure
            elif key == ord('s'):
                self._adjust_exposure(-1)  # Decrease exposure
            return True
        else:
            return False

    def get_setting_ROI(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the ROI region of the given frame.

        Args:
            frame (np.ndarray): The frame to extract the ROI from.

        Returns:
            np.ndarray: The ROI region of the frame.
        """
        return frame[
            self.roi.y:self.roi.y + self.roi.height,
            self.roi.x:self.roi.x + self.roi.width
        ]

    def to_binary(self, image: np.ndarray, otsu: bool = False) -> np.ndarray:
        """
        Convert an image to a binary (black and white) format.

        Args:
            image (np.ndarray): The input image in BGR format.
            otsu (bool, optional): Whether to use Otsu's thresholding. Defaults to False.

        Returns:
            np.ndarray: The binary image resulting from thresholding.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if otsu:
            return cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)[1]

    def get_frame(self) -> Optional[np.ndarray]:  
        """
        Retrieve a frame from either a camera source or a frame queue.

        If the capture source is a camera, read a frame from the camera. If the
        capture source is a queue, attempt to retrieve a frame from the queue.
        If the queue is empty, log a warning and return None.

        Returns:
            Optional[np.ndarray]: A frame if one was successfully read, otherwise None.
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
        self._frame_id += 1
        return frame

    def control_frame_rate(self, start_time: float) -> None:
        """
        Control the frame rate by adding a delay to ensure a consistent frame rate.

        Given a start time, calculate the elapsed time since the start time and the
        desired frame delay. Calculate a delay to add to the current time to ensure
        a consistent frame rate. The minimum delay is 25ms.

        Parameters:
            start_time (float): The start time of the frame processing.
        """
        elapsed_time = time.time() - start_time
        delay = max(0.025, self.frame_delay - elapsed_time)
        if self._frame_id % 60 == 0:
            logger.debug(f"Frame processing time: {elapsed_time:.3f}s, delay: {delay:.3f}s. fps: {1/delay:.2f}")
        time.sleep(delay)

    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        if self.cap and type(self.cap) is cv2.VideoCapture:
            self.cap.release()
        cv2.destroyAllWindows()

