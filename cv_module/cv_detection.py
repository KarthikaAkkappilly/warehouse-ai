#!/usr/bin/env python
# coding: utf-8

# # Warehouse Object Detection
# # Using Sobel Edge Detection + Contour Analysis

# In[1]:


from google.colab import drive
drive.mount('/content/drive')

# Just list what's in your manual folder
get_ipython().system('ls -la "/content/drive/MyDrive/warehouse_ai/cv_module/samples/"')


# In[2]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


# List of sample images
IMAGE_PATHS = [
    "/content/drive/MyDrive/warehouse_ai/cv_module/samples/pallets.jpg",
    "/content/drive/MyDrive/warehouse_ai/cv_module/samples/boxstack.webp",
    "/content/drive/MyDrive/warehouse_ai/cv_module/samples/lengthybox.jpg",
    "/content/drive/MyDrive/warehouse_ai/cv_module/samples/cardboardbox.jpg"
]


# In[ ]:


import cv2 as cv
import numpy as np

def detect_objects(image, area_threshold=10000):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    sobel_x = cv.Sobel(blur, cv.CV_64F, 1, 0, 3)
    sobel_y = cv.Sobel(blur, cv.CV_64F, 0, 1, 3)
    sobel_mag = np.uint8(np.clip(np.sqrt(sobel_x**2 + sobel_y**2), 0, 255))

    _, thresh = cv.threshold(sobel_mag, 30, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    out_img = image.copy()
    objects_info = []

    for c in contours:
        if cv.contourArea(c) < area_threshold:
            continue
        hull = cv.convexHull(c)
        x, y, w, h = cv.boundingRect(hull)
        cx, cy = x + w // 2, y + h // 2

        cv.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(out_img, (cx, cy), 4, (0, 0, 255), -1)
        cv.putText(out_img, f"W:{w} H:{h}", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        objects_info.append({
            "top_left": (x, y),
            "width": w,
            "height": h,
            "center": (cx, cy)
        })

    return out_img, objects_info


# #Explanation
# This Python script performs warehouse object detection using edge detection and contour analysis.
# 1. Each input image is converted to grayscale and blurred using Gaussian Blur to reduce noise.
# 2. Sobel edge detection computes gradients in X and Y directions, giving the edge magnitude of the image.
# 3. The gradient magnitude image is thresholded to produce a binary edge map.
# 4. External contours are extracted from this binary map to identify individual objects.
# 5. For each contour, small noise contours are filtered by area. The convex hull is used to ensure broken edges are connected.
# 6. A bounding rectangle is drawn around each object, the width, height, and center are calculated and displayed.
# 7. The result is visualized using Matplotlib, showing bounding boxes and centers for all detected objects.
# 
# Reason for choosing Sobel over Laplacian or Canny:
# 
# - **Sobel** provides directional gradients (X and Y), allowing control over edge detection orientation.
# - **Laplacian** is a second-order derivative and is very sensitive to noise; it can produce fragmented or weak edges, making contour extraction less reliable.
# - **Canny** is more sophisticated (non-maximum suppression + hysteresis thresholding), but it often detects many small inner edges or thin lines, leading to numerous irrelevant contours in complex warehouse images.
# - Sobel offers a **good balance of simplicity, robustness, and controllable thresholding**, making it easier to extract meaningful contours for large objects like boxes and pallets.
# 
# Bounding boxes, dimensions, and center coordinates are all computed in pixel units.
# This approach can be extended to video feeds or tracking by iterating through frames similarly.
# 
