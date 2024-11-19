import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Function to calculate the Hausdorff distance
def hausdorff_distance(contour1, contour2):
    h1 = directed_hausdorff(contour1, contour2)[0]
    h2 = directed_hausdorff(contour2, contour1)[0]
    return max(h1, h2)

def calculate_score(template_contour, candidate_contour):
    # Hausdorff Distance
    h_distance = hausdorff_distance(template_contour[:, 0, :], candidate_contour[:, 0, :])

    # Normalize by template bounding box diagonal
    x, y, w, h = cv2.boundingRect(template_contour)
    diagonal = np.sqrt(w**2 + h**2)
    normalized_distance = h_distance / diagonal

    # Size ratio (scaling factor)
    template_area = cv2.contourArea(template_contour)
    candidate_area = cv2.contourArea(candidate_contour)
    size_ratio = abs(1 - candidate_area / template_area)

    # Combine the scores (weighted sum)
    combined_score = normalized_distance + 0.5 * size_ratio  # Adjust weights as needed
    return combined_score


cv2.namedWindow('Matched Object', cv2.WINDOW_NORMAL)

# Load the template and target image
template = cv2.imread('data/small/IMG_E0103_zoomed.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('template/sample_1_template.png', cv2.IMREAD_GRAYSCALE)

# Preprocess the template and image
template_edges = cv2.Canny(template, 100, 150)
image_edges = cv2.Canny(image, 100, 150)

# Find contours in both
template_contours, _ = cv2.findContours(template_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_contours, _ = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour in the template is the object
template_contour = max(template_contours, key=cv2.contourArea)

# Accept/Reject Match
threshold = 0.3  # Set based on experiments
best_score = float('inf')
best_contour = None

for contour in image_contours:
    score = calculate_score(template_contour, contour)
    if score < threshold and score < best_score:
        best_score = score
        best_contour = contour

if best_contour is not None:
    print(f"Accepted match with score: {best_score}")
else:
    print("No acceptable match found.")

