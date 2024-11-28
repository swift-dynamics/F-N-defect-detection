import cv2
import numpy as np
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

class ExtractText:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.window_name = "Frame"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.zoom_factor = 1.5
        
    def zoom_in(self, image, zoom_factor=1.5):
        # Get dimensions of the input image
        h, w = image.shape[:2]
        
        # Calculate the size of the crop area based on zoom factor
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
        
        # Calculate the coordinates for the center crop
        start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
        
        # Crop the image to the calculated ROI
        cropped_image = image[start_y:start_y + new_h, start_x:start_x + new_w]
        
        # Resize cropped image back to the original size
        zoomed_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed_image 
    
    def defect_detection(self, image):
        # Use Tesseract to extract text from the isolated white areas
        extracted_text = pytesseract.image_to_string(image, lang='eng+tha')
        output_text = extracted_text.split('\n')
        output_text = [x for x in output_text if x != '']
        
        if output_text and len(output_text) != 4:
            return output_text
        
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
        
        return eroded_image, crop_image
    
    def extract(self):
        while True:
            start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            
            frame = self.zoom_in(frame, zoom_factor=self.zoom_factor)
            
            # BGR to GRAY   
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # gray to binary
            binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            eroded_image, crop_image = self.process_image(binary_image)
            
            # defect detection
            output_text = self.defect_detection(eroded_image)
            
            # visualize the edge contours
            edged = cv2.Canny(gray, 30, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 1) 

            if output_text:
                cv2.putText(frame, "Defect detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No defect", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            fps = time.time() - start
            cv2.putText(frame, f"FPS: {int(1/fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow(self.window_name, frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            key = cv2.waitKey(1) & 0xFF
            # Adjust zoom with keys
            if key == ord('d'):  # Increase zoom
                self.zoom_factor = min(self.zoom_factor + 0.1, 5.0)  # Max zoom level of 5
            elif key == ord('a'):  # Decrease zoom
                self.zoom_factor = max(self.zoom_factor - 0.1, 1.0)  # Min zoom level of 1 (original size)
            elif key == ord('q'):  # Quit
                break
             
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    source = 0
    extractor = ExtractText(source)
    extractor.extract()