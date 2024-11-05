import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

class ExtractText:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.window_name = "Frame"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def defect_detection(self, image):
        # Use Tesseract to extract text from the isolated white areas
        extracted_text = pytesseract.image_to_string(image, lang='eng+tha')
        output_text = [x for x in output_text if x != '']
        
        if len(output_text) != 4:
            return "Defect detected"
        
        return None
    
    def process_image(self, image):
        # crop
        crop_image = image[0:image.shape[0]//2 - 50, 10:image.shape[1]//2]
        
        # Step 4: Apply morphological operations to refine the shapes
        kernel = np.ones((2, 2), np.uint8)
        morph_image = cv2.morphologyEx(crop_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Optional: Additional dilation and erosion to make text more uniform
        dilated_image = cv2.dilate(morph_image, kernel, iterations=1)
        eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
        
        return eroded_image
    
    def extract(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # BGR to GRAY   
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # gray to binary
            binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            eroded_image = self.process_image(binary_image)
            
            # defect detection
            output_text = self.defect_detection(eroded_image)
            
            # visualize the edge contours
            edged = cv2.Canny(gray, 30, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3) 

            if output_text is not None:
                cv2.putText(frame, output_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No defect", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
             
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    source = 0
    extractor = ExtractText(source)
    extractor.extract()