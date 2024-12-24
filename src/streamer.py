from multiprocessing import Queue
import cv2
import time
from typing import Optional
from contextlib import contextmanager
import logging

class CameraStreamer:
    """
    A class to handle camera streaming operations.

    Attributes:
        queue (Queue): A multiprocessing queue for frame storage.
        camera_source: Source input for the camera (device index or video file path).
        fps (int): Desired frames per second for the stream.
    """

    def __init__(self, queue: Queue, camera_source, fps: int = 30):
        """
        Initialize the camera streamer.

        Args:
            queue (Queue): Multiprocessing queue for frame storage.
            camera_source: Camera source input.
            fps (int, optional): Desired frames per second. Defaults to 30.

        Raises:
            ValueError: If the camera cannot be opened.
        """
        self.queue = queue
        self.fps = fps
        self._frame_delay = 1 / fps
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        logging.info("Initializing CameraStreamer with FPS: %d", fps)
        self._initialize_camera(camera_source)

    def _initialize_camera(self, camera_source) -> None:
        """
        Initialize the camera connection.

        Args:
            camera_source: Camera source input.

        Raises:
            ValueError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(camera_source)
        if not self._cap.isOpened():
            logging.error("Cannot open camera source: %s", camera_source)
            raise ValueError(f"Cannot open camera source: {camera_source}")
        logging.info("Camera source initialized successfully: %s", camera_source)

    def start_stream(self) -> None:
        """
        Start the camera stream and begin capturing frames.
        """
        self._running = True
        logging.info("Starting the camera stream.")
        try:
            self._stream_frames()
        except Exception as e:
            logging.error("Error in streaming: %s", e, exc_info=True)
        finally:
            self.stop()

    def _stream_frames(self) -> None:
        """
        Handle the frame streaming loop.
        """
        logging.debug("Entering streaming loop.")
        while self._running:
            success, frame = self._cap.read()
            if not success:
                logging.warning("Failed to capture frame, stopping stream.")
                break

            # Attempt to push the frame into the queue without blocking indefinitely
            try:
                self.queue.put(frame, block=False)
                logging.debug("Frame added to queue. Queue size: %d", self.queue.qsize())
            except Exception:
                logging.warning("Queue is full. Dropping frame.")
            
            time.sleep(self._frame_delay)

    def stop(self) -> None:
        """
        Stop the stream and release resources.
        """
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logging.info("Camera released and streaming stopped.")

    @contextmanager
    def stream_context(self):
        """
        Context manager for handling the camera stream.
        """
        try:
            logging.info("Entering streaming context.")
            yield self
            self.start_stream()
        finally:
            self.stop()
            logging.info("Exiting streaming context.")


def start_stream(queue: Queue, camera_source, fps: int = 30):
    try:
        streamer = CameraStreamer(queue, camera_source, fps)
        streamer.start_stream()
    except Exception as e:
        logging.error(f"Streamer process failed: {e}", exc_info=True)
        raise
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Camera Streamer")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    logging.basicConfig(
            format='%(asctime)s - %(message)s', 
            datefmt='%d-%b-%y %H:%M:%S',
            level=logging.DEBUG if args.debug else logging.INFO
    )

    FRAME_QUEUE = Queue(maxsize=10)
    CAMERA_SOURCE = "data/Relaxing_highway_traffic.mp4"
    FPS = 30
    THRESHOLD = 0.75

    start_stream(FRAME_QUEUE, CAMERA_SOURCE, FPS)
    