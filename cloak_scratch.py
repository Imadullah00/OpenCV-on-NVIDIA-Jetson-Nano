import numpy as np
import cv2

bgr_values = np.array([
    [99, 107, 181],
    [96, 104, 170],
    [94, 101, 172],
    [84, 90, 156],
    [104, 107, 173]
], dtype=np.uint8)

# Convert to HSV (reshape for single-pixel images)
hsv_values = cv2.cvtColor(bgr_values.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

print("HSV values:")
for hsv in hsv_values:
    print(hsv)
