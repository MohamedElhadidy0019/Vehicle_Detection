#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import cv2 
import  matplotlib.pyplot as plt
from time import sleep
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


#Takes two 2D boxes in (x,y,width,height) format
#Returns the area of the intersection box
def area_intersection_box(R1,R2):
    x1,y1,w1,h1 = R1 
    x2,y2,w2,h2 = R2 
    
    if x1 >= (x2+w2): return 0.0
    if x2 >= (x1+w1): return 0.0
    if y1 >= (y2+h2): return 0.0
    if y2 >= (y1+h1): return 0.0
    
    #Width of the intersection box
    w = abs(max(x1,x2) - min(x1+w1,x2+w2))
    #Height of the intersection box
    h = abs(max(y1,y2) - min(y1+h1,y2+h2))
    
    return w*h
    
    


# In[25]:


#Returns the greatest of two boxes by area
def max_box(R1,R2):
    if R1[2]*R1[3] > R2[2]*R2[3]:
        return R1
    
    return R2


# In[26]:


def filter_bounding_boxes(BBs):
    length = len(BBs)
    is_BB_included = [True for i in range(0,length)]
    
    i = 0
    while i < length:
        j = i+1 
        while j < length:
            A = area_intersection_box(BBs[i],BBs[j])
            
            is_BB_included[j] = is_BB_included[j] and (A == 0)
            
            j = j+1
        i = i+1 
    
    new_BBs = [BBs[i] for i in range(0,length) if is_BB_included[i]]
    return new_BBs
            


# In[27]:


def plt_image(img):
    plt.axis("off")
    plt.imshow(img,cmap="gray")
    plt.show()


# In[29]:


##### load a video
video = cv2.VideoCapture('trimmed1.mp4')

# You can set custom kernel size if you want.
kernel = None

# Initialize the background object.
backgroundObject = cv2.createBackgroundSubtractorMOG2(history=120,detectShadows = True)
first_time=True
while True:
    
    # Read a new frame.
    ret, frame = video.read()

    # Check if frame is not read correctly.
    if not ret:
        break

    # Apply the background object on the frame to get the segmented mask. 
    fgmask = backgroundObject.apply(frame)
    #initialMask = fgmask.copy()
    
    # Perform thresholding to get rid of the shadows.
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    #noisymask = fgmask.copy()
    
    # Apply some morphological operations to make sure you have a good mask
    #kernel=np.ones((6,18),np.uint8)
    fgmask = cv2.erode(fgmask,kernel)
    fgmask = cv2.erode(fgmask,kernel)
    
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    fgmask = cv2.dilate(fgmask,kernel)
    if(first_time):
        plt_image(fgmask)
        first_time=False
    # Detect contours in the frame.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the frame to draw bounding boxes around the detected cars.
    frameCopy = frame.copy()
    
    BBs = []
    # loop over each contour found in the frame.
    for cnt in contours:
        # Retrieve the bounding box coordinates from the contour.
        x, y, width, height = cv2.boundingRect(cnt)
        #Thresholding to get rid of false boxes
        if width > 30 and height > 30 :
            BBs.append((x,y,width,height))
                
    BBs = filter_bounding_boxes(BBs)
    for BB in BBs:
        x,y,width,height = BB
        # Draw a bounding box around the car.
        cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
    
    # Extract the foreground from the frame using the segmented mask.
    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
        
    # Stack the original frame, extracted foreground, and annotated frame. 
    stacked = np.hstack((frame, foregroundPart, frameCopy))

    # Display the stacked image with an appropriate title.
    cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked, None, fx=0.5, fy=0.5))
    #cv2.imshow('initial Mask', initialMask)
    #cv2.imshow('Noisy Mask', noisymask)
    #cv2.imshow('Clean Mask', fgmask)

    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xff
    
    # Check if 'q' key is pressed.
    if k == ord('q'):
        plt_image(fgmask)

        # Break the loop.
        break

# Release the VideoCapture Object.
video.release()

# Close the windows.q
cv2.destroyAllWindows()


# In[ ]:




