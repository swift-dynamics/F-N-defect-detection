from multiprocessing import Process, Queue
import time
import logging
import os
import sys
import signal
import dotenv
import argparse
from src.metallic_detector import MetallicDetector
from src.streamer import CameraStreamer
from src.ocr_exp_date import TextExtractor
from src.setting_mode import SettingMode

dotenv.load_dotenv(dotenv_path=".env", override=True)
# dotenv.load_dotenv(dotenv_path='./setting.env', override=True)

# Configuration
CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', "data/Relaxing_highway_traffic.mp4")
FPS = int(os.getenv('FPS', 30))
METALLIC_ALERT_DIRECTORY = str(os.getenv("METALLIC_DEFECTED_ALERT_DIRECTORY", "metallic_defected_images"))
THRESHOLD = float(os.getenv('THRESHOLD', 0.5))
OCR_ALERT_DIRECTORY = str(os.getenv('TEXT_DEFECTED_ALERT_DIRECTORY', "data/alerts"))
TEXT_THRSHOLD = int(os.getenv('TEXT_THRSHOLD', 3))
TEMPLATE_IMAGE = str(os.getenv('TEMPLATE_PATH', None))
# SETTING_ENV_PATH = str(os.getenv('SETTING_ENV_PATH', None))
QSIZE = int(os.getenv('QSIZE', 10))

# Argument Parsing
parser = argparse.ArgumentParser(description="Main-program")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--setting", action="store_true", help="Enable Setting Mode")
parser.add_argument("--main_disp", action="store_true", help="Enable Main Display")
parser.add_argument("--process_disp", action="store_true", help="Enable Process Display")
args = parser.parse_args()

# Logging Configuration
logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG if args.debug else logging.INFO
)
logger = logging.getLogger(__name__)

def broadcaster(input_queue: Queue, output_queues: list[Queue]):
    """
    Broadcast frames from the input queue to multiple output queues.
    """
    while True:
        try:
            frame = input_queue.get(timeout=1)
            for queue in output_queues:
                if not queue.full():
                    queue.put(frame)
        except Exception as e:
            # Handle empty queue or timeout
            print(f"Broadcaster encountered an issue: {e}")
            time.sleep(0.01)

def start_stream(queue: Queue, camera_source, fps: int = 30):
    """
    Start the camera streamer process.
    """
    try:
        logger.info("Starting camera stream process.")
        streamer = CameraStreamer(queue, camera_source, fps)
        streamer.start_stream()
    except Exception as e:
        logger.error(f"Streamer process failed: {e}", exc_info=True)
        raise

def metallic_detector(
        queue: Queue, 
        threshold: float, 
        fps: int,
        alert_directory: str,
        template_image,
        main_window,
        process_window,
        alert_info
    ):
    try:
        logger.info("Starting metallic detector process.")
        detector = MetallicDetector(
            source=queue, 
            fps=fps,
            alert_directory=alert_directory,
            simm_threshold=threshold,
            template_image=template_image,
            main_window=main_window,
            process_window=process_window,
            alert_info=alert_info
        )
        detector.run()
    except Exception as e:
        logger.error(f"Metallic detector process failed: {e}", exc_info=True)
        raise

def exp_date_detector(
        queue: Queue, 
        threshold: int, 
        fps: int,
        alert_directory: str,
        main_window,
        process_window,
        alert_info
    ):
    try:
        logger.info("Starting exp_date_detector process.")
        detector = TextExtractor(
            source=queue, 
            fps=fps,
            alert_directory=alert_directory,
            text_threshold=threshold,
            main_window=main_window,
            process_window=process_window,
            alert_info=alert_info
        )
        detector.run()
    except Exception as e:
        logger.error(f"exp_date_detector process failed: {e}", exc_info=True)
        raise

def terminate_processes(processes):
    """
    Terminate all running processes gracefully.
    """
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
            logger.info(f"Terminated {process.name} with exit code {process.exitcode}")

def main():
    # Run setting mode if specified
    if args.setting:
        logger.info("Entering setting mode.")
        try:
            setting_mode = SettingMode(
                camera_source=CAMERA_SOURCE,
                fps=FPS,
                env_path='.env',
                template_file=TEMPLATE_IMAGE
            )
            setting_mode.run()
        except Exception as e:
            logger.error(f"Setting mode encountered an error: {e}", exc_info=True)
        finally:
            sys.exit(0)

    # Initialize queues
    input_queue = Queue(maxsize=QSIZE)
    output_queue_1 = Queue(maxsize=QSIZE)
    output_queue_2 = Queue(maxsize=QSIZE)
    output_queues = [output_queue_1, output_queue_2]

    # Initialize processes
    processes = [
        Process(target=start_stream, args=(input_queue, CAMERA_SOURCE, FPS), name='start_stream_process'),
        Process(target=broadcaster, args=(input_queue, output_queues), name='broadcaster_process'),
        Process(
            target=metallic_detector, 
            args=(
                output_queue_1, 
                THRESHOLD, 
                FPS, 
                METALLIC_ALERT_DIRECTORY, 
                TEMPLATE_IMAGE,
                "Metallic-main-display" if args.main_disp else None, 
                "Metallic-process-display" if args.process_disp else None, 
                "metallic-defected"
            ), 
            name='milk_corton_detector_process'
        ),
        Process(
            target=exp_date_detector, 
            args=(
                output_queue_2, 
                TEXT_THRSHOLD, 
                FPS, 
                OCR_ALERT_DIRECTORY,
                "OCR-main-display" if args.main_disp else None, 
                "OCR-process-display" if args.process_disp else None, 
                "text-defected"
            ), 
            name='exp_date_detector_process'
        ),
    ]

    try:
        for process in processes:
            process.start()
            logger.info(f"Started {process.name}")

        # Handle termination signals
        signal.signal(signal.SIGINT, lambda sig, frame: terminate_processes(processes))
        signal.signal(signal.SIGTERM, lambda sig, frame: terminate_processes(processes))

        # Monitor processes
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
