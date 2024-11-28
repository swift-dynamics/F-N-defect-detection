import cv2
import numpy as np

class MetallicDetector:
    def __init__(self):
        pass
    
    def segmentation(self, image, kernel_size=(55, 55)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, kernel_size, 0)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def edge_detection(self, binary_image, threshold1=50, threshold2=150):
        edges = cv2.Canny(binary_image, threshold1, threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
        else:
            max_contour = None
        return contours, max_contour        
    
    def detect(self, image, lower=None, upper=None):
        if lower is None:
            lower = [70, 30, 70]
        if upper is None:
            upper = [100, 150, 120]
        
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the specific color range
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        
        # Bitwise AND to extract the color from the original image
        result = cv2.bitwise_and(image, image, mask=mask)
                
        # Segmentation
        result_binary = self.segmentation(result, kernel_size=(35, 35))
        
        # Edge detection
        contours, max_contour = self.edge_detection(result_binary)
        
        return contours, max_contour
    
    def run(self, source=0, offset=0, kernel_size=(85, 85)):
        cv2.namedWindow('metallic_detector', cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # Step 1: Segmentation
            binary = self.segmentation(frame, kernel_size=kernel_size)
            
            # Step 2: Edge detection
            edges, max_contour = self.edge_detection(binary)
            if max_contour is None:
                print("max_contour is none", max_contour)
                roi_frame = frame.copy()
            else:                
                # Step 3: Get the bounding box of the largest contour
                x, y, w, h = cv2.boundingRect(max_contour)
                
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Step 4: Crop the image
                cropped_frame = frame[y+offset:y+h-offset, x+offset:x+w-offset]
                
                if cropped_frame.size == 0:
                    roi_frame = frame.copy()
                else:
                    # Step 5: Detect the metallic object
                    contours, metallic_contour = self.detect(cropped_frame)
                    if metallic_contour is not None:
                        
                        roi_frame = cv2.drawContours(frame.copy(), [metallic_contour], -1, (0, 255, 0), 3)
                    else:
                        print("No metallic detected!", cropped_frame.size)
                        roi_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 3)
                        # roi_frame = frame.copy()
            
            # Display the result
            cv2.imshow('metallic_detector', roi_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    metallic_detector = MetallicDetector()
    metallic_detector.run(source=0, offset=0)
