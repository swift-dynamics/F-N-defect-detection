from multiprocessing import Process, Queue
from typing import Optional
import cv2
import signal
import sys

from src.metallic_detector import MetallicDetector
from src.streamer import CameraStreamer
from src.ocr_exp_date import ExtractText

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print("\nSignal received. Initiating shutdown...")
    sys.exit(0)

def start_camera(frame_queue: Queue, camera_source: str) -> None:
    """Start the camera streaming process.
    
    Args:
        frame_queue: Multiprocessing queue for frame storage
        camera_source: Path to video file or camera index
    """
    try:
        streamer = CameraStreamer(frame_queue, camera_source)
        streamer.start_stream()
    except Exception as e:
        print(f"Error in camera stream: {e}")
    finally:
        if 'streamer' in locals():
            streamer.stop()

def main():
    """Main function to orchestrate the detection system."""
    # Constants
    FRAME_QUEUE_SIZE = 10
    CONFIDENCE_THRESHOLD = 0.7
    FPS_LIMIT = 30
    SOURCE = "data/Relaxing_highway_traffic.mp4"

    # Initialize shared queue for frames
    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    processes = []

    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize camera stream process
        camera_process = Process(
            target=start_camera,
            args=(frame_queue, SOURCE),
            name="CameraProcess"
        )
        processes.append(camera_process)

        # Initialize Metallic Detector process
        detector_process = Process(
            target=MetallicDetector,
            args=(frame_queue, CONFIDENCE_THRESHOLD, FPS_LIMIT),
            name="DetectorProcess"
        )
        processes.append(detector_process)

        # Start all processes
        for process in processes:
            process.start()
            print(f"Started process: {process.name}")

        # Wait for processes to complete
        for process in processes:
            process.join()

    except Exception as e:
        print(f"Error in main process: {e}")
    finally:
        # Cleanup
        for process in processes:
            if process.is_alive():
                print(f"Terminating process: {process.name}")
                process.terminate()
                process.join()
        
        # Clear the queue
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass

if __name__ == "__main__":
    main()
