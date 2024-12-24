from multiprocessing import Process, Queue
import time
import logging
import os
import sys
import signal
import dotenv
from src.metallic_detector import MetallicDetector
from src.streamer import CameraStreamer

dotenv.load_dotenv(override=True)

CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', 0)
FPS = os.getenv('FPS', 30)
THRESHOLD = os.getenv('THRESHOLD', 0.5)
SETTING_ENV_PATH = os.getenv('SETTING_ENV_PATH', None)
IMAGE_TEMPLATE = os.getenv('IMAGE_TEMPLATE', None)
FRAME_QUEUE = Queue(maxsize=10)

def start_stream(queue: Queue, camera_source, fps: int = 30):
    try:
        streamer = CameraStreamer(queue, camera_source, fps)
        streamer.start_stream()
    except Exception as e:
        logging.error(f"Streamer process failed: {e}", exc_info=True)
        raise

def milk_corton_detector(queue: Queue, threshold: float = 0.5, fps: int = 30):
    try:
        detector = MetallicDetector(queue, threshold, fps)
        detector.run()
    except Exception as e:
        logging.error(f"Detector process failed: {e}", exc_info=True)
        raise

def terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
        logging.info(f"Terminated {process.name} with exit code {process.exitcode}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setting Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--setting", action="store_true", help="Enable Setting Mode")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.DEBUG if args.debug else logging.INFO
    )
    # If setting mode is enabled, run the setting mode
    if args.setting:
        from src.setting_mode import SettingMode
        setting_mode = SettingMode(camera_source=CAMERA_SOURCE, fps=FPS, env_path=SETTING_ENV_PATH, image_template=IMAGE_TEMPLATE)
        setting_mode.run()
        sys.exit(0)

    processes = []
    try:
        # Initialize and start processes
        start_stream_process = Process(target=start_stream, args=(FRAME_QUEUE, CAMERA_SOURCE, FPS), name='start_stream_process')
        milk_corton_detector_process = Process(target=milk_corton_detector, args=(FRAME_QUEUE, THRESHOLD, FPS), name='milk_corton_detector_process')
        
        processes.extend([start_stream_process, milk_corton_detector_process])
        for process in processes:
            process.start()
            logging.info(f"Started {process.name}")

        # Wait for termination signal
        signal.signal(signal.SIGINT, lambda sig, frame: terminate_processes(processes))
        signal.signal(signal.SIGTERM, lambda sig, frame: terminate_processes(processes))

        while any(process.is_alive() for process in processes):
            time.sleep(1)

    except Exception as e:
        logging.error(f"Main process encountered an error: {e}", exc_info=True)
        terminate_processes(processes)
    finally:
        terminate_processes(processes)
        sys.exit(0)
