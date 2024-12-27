import os
from datetime import datetime, timedelta
import logging
from typing import Optional
import cv2
import numpy as np
import dotenv
from services import Minio

dotenv.load_dotenv(override=True)

ALERT_DEBOUNCE_SECONDS = int(os.getenv('ALERT_DEBOUNCE_SECONDS', 10))
BUCKET_NAME = str(os.getenv("BUCKET_NAME"))

logger = logging.getLogger(__name__)

class AlertProcessBase:
    """
    Base class for alert processing.
    """

    def __init__(self, save: bool, minio: bool) -> None:
        """
        Initialize the alert process base.

        Args:
            save_to_local (bool): Whether to save alerts to local file system.
            save_to_minio (bool): Whether to save alerts to MinIO.
        """
        self.save_to_local = save
        self.save_to_minio = minio
        logger.info("Alert process base initialized.")
        logger.info(f"Alert debounce seconds: {ALERT_DEBOUNCE_SECONDS}")

    def setup_alert(self, alert_directory: Optional[str], alert_info: str) -> None:
        """
        Set up the alert process.

        Args:
            alert_directory (str | None): The directory to save alerts to.
            alert_info (str): The type of alert.
        """
        self.alert_count = 0
        self.alerted = False
        self.last_alert_time = datetime.min
        self.alert_debounce = timedelta(seconds=ALERT_DEBOUNCE_SECONDS)
        if alert_directory:
            self.alert_info = alert_info
            self.root_dir = os.path.join(os.getcwd(), alert_directory)
            if self.save_to_local:
                os.makedirs(self.root_dir, exist_ok=True)
        else:
            logger.warning("Alert directory is not provided.")

    def trigger_alert(self, frame: np.ndarray, info: Optional[str] = None) -> None:
        """
        Process the extracted text and take appropriate action.

        Args:
            frame (np.ndarray): The frame from the video stream.
            info (str | None): Additional information about the alert.
        """
        current_time = datetime.now()
        if not self.alerted and (current_time - self.last_alert_time > self.alert_debounce):
            # Trigger alert
            self.alerted = True
            self.last_alert_time = current_time
            # Count
            self.alert_count += 1

            logger.info(f"Alert triggered at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{info}") if info else None

            # Save defected image
            timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            defect_image_path = os.path.join(self.root_dir, f"{timestamp}_{self.alert_info}.jpg")

            if self.save_to_local:
                cv2.imwrite(defect_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"Text image saved to: {defect_image_path}")

            # Upload to minio
            if self.save_to_minio:
                object_name = f"{self.alert_info}/{timestamp}.jpg"
                Minio.upload_image(defect_image_path, BUCKET_NAME, object_name)
