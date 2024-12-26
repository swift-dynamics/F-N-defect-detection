from dataclasses import dataclass
from typing import Optional
import time
import cv2
import logging
from filelock import FileLock
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

@dataclass
class ROICoordinates:
    x: int
    y: int
    width: int
    height: int

class CreateTemplate:
    def __init__(self, camera_source: str | int = 0, fps: int = 30, window_name: str = 'setting_mode', 
                 env_path: str = None, template_file: str = None) -> None:
        logger.debug(f"Initializing CreateTemplate with camera_source={camera_source}, fps={fps}, window_name={window_name}")
        self.window_name = window_name
        self._setup_camera(camera_source)
        self._setup_display_parameters(fps)
        self.roi: Optional[ROICoordinates] = None
        self.env_path = env_path 
        self.template_file = template_file

        if env_path is None or template_file is None:
            logger.warning("Environment path or image template not provided. ROI selection will not be saved.")
            raise ValueError("Environment path or image template not provided.")
        
    def _setup_camera(self, camera_source: str | int) -> None:
        """Initialize and validate camera connection."""
        logger.debug(f"Setting up camera with source: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            logger.debug("Failed to open camera")
            raise ValueError("Failed to access camera/video source")
            
        ret, self.frame = self.cap.read()
        if not ret:
            logger.debug("Failed to read initial frame")
            raise ValueError("Failed to read frame from camera/video source")
            
        self.frame_height, self.frame_width = self.frame.shape[:2]
        logger.debug(f"Frame dimensions: {self.frame_width}x{self.frame_height}")

    def _setup_display_parameters(self, fps: int) -> None:
        """Setup display window and frame rate parameters."""
        logger.debug(f"Setting up display parameters with fps: {fps}")
        self.frame_delay = 1.0 / fps
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def setting_up(self) -> None:
        """Allow user to select ROI using GUI."""
        logger.debug("Starting ROI selection")
        x, y, w, h = cv2.selectROI(
            self.window_name, 
            self.frame, 
            fromCenter=False, 
            showCrosshair=True
        )
        self.roi = ROICoordinates(x, y, w, h)
        logger.debug(f"ROI selected with coordinates: x={x}, y={y}, width={w}, height={h}")

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events for ROI selection."""
        if event == cv2.EVENT_RBUTTONDOWN:
            logger.info("Right mouse button clicked, initiating ROI selection")
            logger.debug(f"Mouse click at coordinates: x={x}, y={y}")
            self.setting_up()
            logger.info(f"ROI selected: {self.roi}")
            # Save ROI coordinates to an env file
            self.__save_roi(self.roi, self.env_path)
            # Save cropped template to a file
            if self.template_file is not None:
                logger.debug("Cropping frame for template")
                roi_frame = self.frame[
                    self.roi.y:self.roi.y + self.roi.height,
                    self.roi.x:self.roi.x + self.roi.width
                ]
                self.__save_template_file(roi_frame, self.template_file, self.env_path)
            
    def __save_roi(self, roi: ROICoordinates, file_path: str) -> None:
        """Save ROI coordinates to an env file, preserving static values, using FileLock."""
        logger.debug(f"Saving ROI coordinates to file: {file_path}")
        
        lock_path = f"{file_path}.lock"  # Create a lock file path
        lock = FileLock(lock_path)  # Create a FileLock instance

        with lock:  # Acquire the lock
            # Load existing static values from the .env file
            existing_values = dotenv_values(file_path)
            
            # Update ROI values
            existing_values.update({
                "X": roi.x,
                "Y": roi.y,
                "WIDTH": roi.width,
                "HEIGHT": roi.height,
            })

            # Write back all values to the .env file
            with open(file_path, 'w') as f:
                for key, value in existing_values.items():
                    f.write(f"{key}={value}\n")

            logger.info(f"ROI coordinates merged and saved to {file_path}")
    
    def __save_template_file(self, template_file: str, file_path: str, env_path: str) -> None:
        """Save cropped template to a file, preserving static values, using FileLock."""
        logger.debug(f"Saving image template to file: {file_path}")
        
        lock_path = f"{env_path}.lock"  # Create a lock file path for the environment file
        lock = FileLock(lock_path)  # Create a FileLock instance

        with lock:  # Acquire the lock
            # Load existing static values from the .env file
            existing_values = dotenv_values(env_path)
            
            # Add the template path
            existing_values["TEMPLATE_PATH"] = file_path

            # Save the cropped template to the specified file
            if template_file.size > 0:
                cv2.imwrite(file_path, template_file, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
                # Write back all values to the .env file
                with open(env_path, 'w') as f:
                    for key, value in existing_values.items():
                        f.write(f"{key}={value}\n")
                
                logger.info(f"Image template saved to {file_path}")
            else:
                logger.error("Failed to save image template.")

    def _draw_roi(self, window_name='cropped_frame') -> None:
        """Draw ROI rectangle on frame if ROI is set."""
        if self.roi and self.roi.height > 0 and self.roi.width > 0:
            logger.debug("Drawing ROI rectangle on frame")
            cv2.rectangle(
                self.frame,
                (self.roi.x, self.roi.y),
                (self.roi.x + self.roi.width, self.roi.y + self.roi.height),
                (0, 255, 0),
                2
            )
            roi_frame = self.frame[
                self.roi.y:self.roi.y + self.roi.height,
                self.roi.x:self.roi.x + self.roi.width
            ]
            cv2.imshow(window_name, roi_frame)
        else:
            logger.debug("No ROI selected")

    def _process_frame(self) -> bool:
        """Process a single frame from the camera."""
        logger.debug("Processing new frame")
        ret, self.frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera/video source")
            return False
        return True

    def run(self) -> None:
        """Main loop for running the setting mode."""
        logger.debug("Starting main loop")
        try:
            while True:
                start_time = time.time()
                
                cv2.setMouseCallback(self.window_name, self._mouse_callback)
                
                if not self._process_frame():
                    break
                    
                self._draw_roi()

                cv2.imshow(self.window_name, self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.debug("Quit signal received")
                    break
                    
                # Control frame rate
                elapsed_time = time.time() - start_time
                delay = max(0.01, self.frame_delay - elapsed_time)
                time.sleep(delay)
                logger.debug(f"Frame processing time: {elapsed_time:.3f}s, delay: {delay:.3f}s, fps: {1/delay:.2f}")

        except KeyboardInterrupt:
            logger.info("Setting mode interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources and close windows."""
        logger.debug("Cleaning up resources")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setting Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logger")
    args = parser.parse_args()
    
    logger.basicConfig(
            format='%(asctime)s - %(message)s', 
            datefmt='%d-%b-%y %H:%M:%S',
            level=logger.DEBUG if args.debug else logger.INFO
    )

    try:
        setting_mode = CreateTemplate(camera_source="data/videos/Relaxing_highway_traffic.mp4", env_path=".env", template_file="data/defected_templates/large_milk_carton_template.png")
        setting_mode.run()
    except ValueError as e:
        logger.error(f"Failed to initialize CreateTemplate: {e}")
