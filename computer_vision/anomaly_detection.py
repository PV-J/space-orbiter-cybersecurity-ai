import cv2

# Load a sample orbital image (replace 'orbital_view.jpg' with actual image path)
image = cv2.imread('orbital_view.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold to detect bright anomalies (potential threats or debris)
_, anomaly_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# Find contours of detected anomalies
contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Highlight anomalies on original image
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imwrite('anomaly_detected.jpg', output)
print("Anomaly detection complete. Check 'anomaly_detected.jpg'.")
