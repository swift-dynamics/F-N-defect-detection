import os
from multiprocessing import Queue
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Union
import cv2
import numpy as np
import dotenv
from services import Minio

dotenv.load_dotenv(override=True)

ALERT_DEBOUNCE_SECONDS = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))
BUCKET_NAME = str(os.getenv("BUCKET_NAME"))

logger = logging.getLogger(__name__)

class AlertProcessBase:
    def __init__(self, save: bool, minio: bool):
        self.save_to_local = save
        self.save_to_minio = minio
        self.alert_count = 0
        logger.info("Alert process base initialized.")
        logger.info(f"Alert debounce seconds: {ALERT_DEBOUNCE_SECONDS}")
    
    def setup_alert(self, alert_directory, alert_info) -> None:
        if alert_directory:
            self.alert_info = alert_info
            self.alerted = False
            self.last_alert_time = datetime.min
            self.alert_debounce = ALERT_DEBOUNCE_SECONDS
            
            self.root_dir = os.path.join(os.getcwd(), alert_directory)
            if self.save_to_local:
                os.makedirs(self.root_dir, exist_ok=True)
        else:
            logger.warning("Alert directory is not provided.")

    def trigger_alert(self, frame):
        """
        Process the extracted text and take appropriate action.
        """
        current_time = datetime.now()
        if not self.alerted or (current_time - self.last_alert_time > timedelta(seconds=self.alert_debounce)):
            # Trigger alert
            self.alerted = True
            self.last_alert_time = current_time
            # Count
            self.alert_count += 1
            
            logger.info(f"Alert triggered at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Save defected image
            timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            defect_image_path = os.path.join(self.root_dir, f"{timestamp}_{self.alert_info}.jpg")
            
            if self.save_to_local:
                cv2.imwrite(defect_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"Text image saved to: {defect_image_path}")

            # Upload to minio
            if self.save_to_minio:
                object_name = f"metallic-defected/{timestamp}_{self.alert_info}.jpg"
                Minio.upload_image(defect_image_path, BUCKET_NAME, object_name)
            
        else:
            logger.debug("Alert suppressed due to debounce logic.")