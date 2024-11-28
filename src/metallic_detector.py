import cv2
import numpy as np

class MetallicDetector:
    def __init__(self,):
        pass
    
    def segmentation(self, image, kernel_size=(75, 75)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, kernel_size, 0)
        ret, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def edge_detection(self, binary_image, threshold1=50, threshold2=150):
        edges = cv2.Canny(binary_image, threshold1, threshold2)
        contours, hierarchy = cv2.findContours(edges,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea)
        return contours, max_contour        
        
    def detect(self, image, lower: list=None, upper: list=None):
        if lower is None:
            lower = [70, 30, 70]
        if upper is None:
            upper = [255, 150, 120]
        
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the specific color range
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        
        # Bitwise AND to extract the color from the original image
        result =  cv2.bitwise_and(image, image, mask=mask)
                
        # segmentation
        result_binary = self.segmentation(result, kernel_size=(55, 55))
        
        # edge detection
        contours, max_contour = self.edge_detection(result_binary)
        
        return contours, max_contour
    
    def run(self, source=0, offset=0):
        cv2.namedWindow('metallic_detector', cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Step 1: Segmentation
            binary = self.segmentation(frame)
            
            # Step 2: Edge detection
            _, max_contour = self.edge_detection(binary)
            
            # Step 3: Get the bounding box of the template contour
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Step 4: Crop the image based on the bounding box
            cropped_frame = frame[y+offset:y+h-offset, x+offset:x+w-offset]
            
            # Step 5: Detect the metallic object
            _, metallic_contour = self.detect(cropped_frame)
            if metallic_contour is None:
                continue
            
            roi_frame = cv2.drawContours(frame.copy(), [metallic_contour], -1, (0, 255, 0), 3)
            cv2.imshow('metallic_detector', roi_frame[:,:,::-1])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows() 
        
if __name__ == "__main__":
    metallic_detector = MetallicDetector()
    metallic_detector.run(source=0, offset=0)