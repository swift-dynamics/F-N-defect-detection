from multiprocessing import Process, Queue
import time
import logging
import os
import sys
import signal
import dotenv
import argparse
from typing import List, Optional, Union
from src import HistogramComparator, CameraStreamer, MFDEXDInspector, CreateTemplate

dotenv.load_dotenv(override=True)

# Configuration
CAMERA_SOURCE: str | int = int(os.getenv('CAMERA_SOURCE')) if os.getenv('CAMERA_SOURCE').isdigit() else os.getenv('CAMERA_SOURCE')
FPS: int = int(os.getenv('FPS', 30))
METALLIC_ALERT_DIRECTORY: str = str(os.getenv("METALLIC_DEFECTED_ALERT_DIRECTORY", "metallic_defected_images"))
THRESHOLD: float = float(os.getenv('THRESHOLD', 0.5))
OCR_ALERT_DIRECTORY: str = str(os.getenv('TEXT_DEFECTED_ALERT_DIRECTORY', "data/alerts"))
TEXT_THRSHOLD: int = int(os.getenv('TEXT_THRSHOLD', 3))
TEMPLATE_IMAGE: Optional[str] = os.getenv('TEMPLATE_PATH', None)
QSIZE: int = int(os.getenv('QSIZE', 10))

# Argument Parsing
parser = argparse.ArgumentParser(description="Main-program")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--setting", action="store_true", help="Enable Template")
parser.add_argument("--main_disp", action="store_true", help="Enable Main Display")
parser.add_argument("--process_disp", action="store_true", help="Enable Process Display")
parser.add_argument("--save", action="store_true", help="Enable Save")
parser.add_argument("--minio", action="store_true", help="Enable MinIO")
args = parser.parse_args()

# Logging Configuration
logging.basicConfig(
    format="{asctime} - [{levelname:^7}] - {module:<20} - {message}",
    datefmt='%Y-%m-%d %H:%M:%S',
    style="{",  # Enable `{}` style formatting
    level=logging.DEBUG if args.debug else logging.INFO
)
logger = logging.getLogger(__name__)

def broadcaster(input_queue: Queue, output_queues: List[Queue]) -> None: 
    """
    Broadcast frames from the input queue to multiple output queues.

    Args:
        input_queue (Queue): Input queue to read frames from.
        output_queues (List[Queue]): List of output queues to write frames to.

    Raises:
        Exception: If the broadcaster encounters an error.
    """
    while True:
        try:
            frame = input_queue.get(timeout=1)
            for queue in output_queues:
                if not queue.full():
                    queue.put(frame)
        except Exception as e:
            logger.error(f"Broadcaster encountered an issue: {e}", exc_info=True)
            raise

def start_stream(queue: Queue, camera_source: Union[str, int], fps: int = 30) -> None:
    """
    Initialize and start the camera streaming process.

    Args:
        queue (Queue): A multiprocessing queue for storing frames.
        camera_source (Union[str, int]): The source input for the camera, either a device index or video file path.
        fps (int, optional): The desired frames per second for the camera stream. Defaults to 30.

    Raises:
        Exception: If the camera streamer process encounters an error.
    """
    try:
        logger.info("Starting camera stream process.")
        streamer = CameraStreamer(queue, camera_source, fps)
        streamer.start_stream()
    except Exception as e:
        logger.error(f"Streamer process failed: {e}", exc_info=True)
        raise

def metallic_detector(queue: Queue, threshold: float, fps: int, alert_directory: str, 
                      template_image: Optional[str], main_window: Optional[str], 
                      process_window: Optional[str], alert_info: str, save: bool, 
                      minio: bool) -> None:
    """
    Run the metallic detector process.

    Args:
        queue (Queue): A multiprocessing queue for storing frames.
        threshold (float): Similarity threshold for detection.
        fps (int): Frames per second for processing.
        alert_directory (str): Directory to save alerts.
        template_image (Optional[str]): Path to the template image.
        main_window (Optional[str]): Main display window name.
        process_window (Optional[str]): Process display window name.
        alert_info (str): Alert information.
        save (bool): Whether to save alerts locally.
        minio (bool): Whether to save alerts to MinIO.

    Raises:
        Exception: If the metallic detector process encounters an error.
    """
    try:
        logger.info("Starting metallic detector process.")
        detector = HistogramComparator(
            source=queue, 
            fps=fps,
            alert_directory=alert_directory,
            simm_threshold=threshold,
            template_image=template_image,
            main_window=main_window,
            process_window=process_window,
            alert_info=alert_info,
            save=save,
            minio=minio
        )
        detector.run()
    except Exception as e:
        logger.error(f"Metallic detector process failed: {e}", exc_info=True)
        raise

def exp_detector(queue: Queue, threshold: int, fps: int, alert_directory: str, 
                 main_window: Optional[str], process_window: Optional[str], 
                 alert_info: str, save: bool, minio: bool) -> None:
    """
    Run the expiration detector process.

    Args:
        queue (Queue): A multiprocessing queue for storing frames.
        threshold (int): Text threshold for detection.
        fps (int): Frames per second for processing.
        alert_directory (str): Directory to save alerts.
        main_window (Optional[str]): Main display window name.
        process_window (Optional[str]): Process display window name.
        alert_info (str): Alert information.
        save (bool): Whether to save alerts locally.
        minio (bool): Whether to save alerts to MinIO.

    Raises:
        Exception: If the expiration detector process encounters an error.
    """
    try:
        logger.info("Starting expiration detector process.")
        detector = MFDEXDInspector(
            source=queue, 
            fps=fps,
            alert_directory=alert_directory,
            text_threshold=threshold,
            main_window=main_window,
            process_window=process_window,
            alert_info=alert_info,
            save=save,
            minio=minio
        )
        detector.run()
    except Exception as e:
        logger.error(f"Expiration detector process failed: {e}", exc_info=True)
        raise

def terminate_processes(processes: List[Process]) -> None:
    """
    Terminate all running processes gracefully.

    Args:
        processes (List[Process]): List of processes to terminate.
    """
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
            logger.info(f"Terminated {process.name} with exit code {process.exitcode}")

def init_process(input_queue: Queue, output_queues: List[Queue]) -> List[Process]:
    """
    Initialize the processes for the metallic and expiration detectors.

    Args:
        input_queue (Queue): The input queue to read frames from.
        output_queues (List[Queue]): A list of output queues to write frames to.

    Returns:
        List[Process]: List of initialized processes.
    """
    processes = [
        Process(target=start_stream, args=(input_queue, CAMERA_SOURCE, FPS), name='start_stream_process'),
        Process(target=broadcaster, args=(input_queue, output_queues), name='broadcaster_process'),
        Process(
            target=metallic_detector, 
            args=(
                output_queues[0], 
                THRESHOLD, 
                FPS, 
                METALLIC_ALERT_DIRECTORY, 
                TEMPLATE_IMAGE,
                "Metallic-main-display" if args.main_disp else None, 
                "Metallic-process-display" if args.process_disp else None, 
                "metallic-defected",
                args.save,
                args.minio
            ), 
            name='milk_corton_detector_process'
        ),
        Process(
            target=exp_detector, 
            args=(
                output_queues[1], 
                TEXT_THRSHOLD, 
                FPS, 
                OCR_ALERT_DIRECTORY,
                "OCR-main-display" if args.main_disp else None, 
                "OCR-process-display" if args.process_disp else None, 
                "text-defected",
                args.save,
                args.minio
            ), 
            name='exp_detector_process'
        ),
    ]
    return processes

def main() -> None:
    """
    Main function to run the application.
    """
    if args.setting:
        logger.info("Entering setting mode.")
        try:
            template_creator = CreateTemplate(
                camera_source=CAMERA_SOURCE, fps=FPS, window_name='setting_mode',
                env_path='.env', template_file=TEMPLATE_IMAGE
            )
            template_creator.run()
        except Exception as e:
            logger.error(f"Setting mode encountered an error: {e}", exc_info=True)
        finally:
            sys.exit(0)

    input_queue = Queue(maxsize=QSIZE)
    output_queue_1 = Queue(maxsize=QSIZE)
    output_queue_2 = Queue(maxsize=QSIZE)
    output_queues = [output_queue_1, output_queue_2]

    processes = init_process(input_queue, output_queues)

    try:
        for process in processes:
            process.start()
            logger.info(f"Started {process.name}")

        signal.signal(signal.SIGINT, lambda sig, frame: terminate_processes(processes))
        signal.signal(signal.SIGTERM, lambda sig, frame: terminate_processes(processes))

        while any(process.is_alive() for process in processes):
            time.sleep(1)

    except Exception as e:
        logger.error(f"Main process encountered an error: {e}", exc_info=True)
        terminate_processes(processes)
    finally:
        terminate_processes(processes)
        logger.info("All processes terminated. Exiting program.")
        sys.exit(0)

if __name__ == "__main__":
    main()

