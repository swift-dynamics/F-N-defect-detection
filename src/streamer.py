from multiprocessing import Process, Queue
import cv2
import time
from typing import Optional
from contextlib import contextmanager

class CameraStreamer:
    """A class to handle camera streaming operations.
    
    Attributes:
        queue (Queue): A multiprocessing queue for frame storage
        camera_source: Source input for the camera (can be device index or video file path)
        fps (int): Desired frames per second for the stream
    """
    
    def __init__(self, queue: Queue, camera_source, fps: int = 30):
        """Initialize the camera streamer.
        
        Args:
            queue (Queue): Multiprocessing queue for frame storage
            camera_source: Camera source input
            fps (int, optional): Desired frames per second. Defaults to 30.
        
        Raises:
            ValueError: If camera cannot be opened
        """
        self.queue = queue
        self.fps = fps
        self._frame_delay = 1 / fps
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._initialize_camera(camera_source)

    def _initialize_camera(self, camera_source) -> None:
        """Initialize the camera connection.
        
        Args:
            camera_source: Camera source input
            
        Raises:
            ValueError: If camera cannot be opened
        """
        self._cap = cv2.VideoCapture(camera_source)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open camera source: {camera_source}")

    def start_stream(self) -> None:
        """Start the camera stream and begin capturing frames."""
        self._running = True
        try:
            self._stream_frames()
        except Exception as e:
            print(f"Error in stream: {str(e)}")
        finally:
            self.stop()

    def _stream_frames(self) -> None:
        """Handle the frame streaming loop."""
        while self._running:
            success, frame = self._cap.read()
            if not success:
                print("Failed to capture frame")
                break
            
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                
            if not self.queue.full():
                self.queue.put(frame)
            
            time.sleep(self._frame_delay)
        
    def stop(self) -> None:
        """Stop the stream and release resources."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @contextmanager
    def stream_context(self):
        """Context manager for handling the camera stream."""
        try:
            yield self
            self.start_stream()
        finally:
            self.stop()
