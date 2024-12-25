import os
from multiprocessing import Queue
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Union
import cv2
import numpy as np
import dotenv

logger = logging.getLogger(__name__)

dotenv.load_dotenv(override=True)

ALERT_DEBOUNCE_SECONDS = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))

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