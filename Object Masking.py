import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/content/drive/MyDrive/Project 3 Data/motherboard_image.JPEG'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# First threshold and contouring
_, thresholded = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
inverted_thresholded = cv2.bitwise_not(thresholded)
contours, _ = cv2.findContours(inverted_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
mask = np.zeros_like(image)
cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
masked_image0 = cv2.bitwise_and(image, mask)

# Second threshold and contouring
min_contour_area = 100
_, thresholded = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
mask = np.zeros_like(image)
cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
masked_image = cv2.bitwise_and(masked_image0, mask)

# Plotting the images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Masked Image
plt.subplot(1, 2, 2)
plt.title('Masked Image')
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Project 3 Data/masked_image.png')
plt.show()
