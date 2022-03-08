import cv2
import numpy as np
from numpy.core.arrayprint import printoptions
import skimage.io as io

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('trimmed1.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

picList=[]
# Read until video is completed
i=200
while(cap.read()):
  # Capture frame-by-frame
  i-=1
  if(i==0):
    break

  ret, frame = cap.read()
  if ret == True:
    if(i==2):
      print(frame.shape)
    picList.append(frame)
    # Display the resulting frame
    #cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
# cap.release()

# Closes all the frames
# cv2.destroyAllWindows()

arr = np.asarray(picList)

print(arr.shape)

meanimage=np.sum(arr,axis=0)//len(picList)
print(meanimage.shape)
io.imshow(meanimage)
print(meanimage)
#cv2.imshow(meanimage)
cv2.imshow("lol", meanimage)
  
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()

