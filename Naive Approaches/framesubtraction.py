import cv2
import numpy as np

# specify video name
videoName = 'trimmed3'
videoWidth = 650
videoHight = 650
# empty list to store the frames
col_images=[]
#set colors
color1 = (0, 255, 255);  color2 = (255, 0, 255);  color3 = (255, 255, 0)
# Opens the Video file
cap = cv2.VideoCapture(videoName + '.mp4')
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (videoWidth, videoHight))
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # append the frames to the list
    col_images.append(frame)
    #cv2.imwrite('inputFrames/frame' + str(i) + '.jpg', frame)
    i += 1
cap.release()
print('the video is fully read!')
########################################################################################################################
# kernel for image dilation
kernel = np.ones((4, 4), np.uint8)
# font style
font = cv2.FONT_HERSHEY_SIMPLEX
# list to save the ouput frames
outputFrames = []
for i in range(len(col_images) - 1):
    cv2.putText(col_images[i], "frame{}".format(i), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color3, 1)
    # drawing a line as a border
    cv2.line(col_images[i], (0, videoHight // 2), (videoWidth, videoHight // 2), color3, 3)
    # frame differencing
    grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(col_images[i + 1], cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)
    # image thresholding
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
    # image dilation
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # shortlist contours
    for cntr in contours:
        contours_poly = cv2.approxPolyDP(cntr, 3, True)
        x, y, w, h = cv2.boundingRect(contours_poly)
        area = cv2.contourArea(contours_poly)
        if y < 400 or area < 700 or area > 1200:
            continue
        #print(i, area)
        #cv2.drawContours(col_images[i], cntr, -1, color1)
        cv2.rectangle(col_images[i], (int(x), int(y)), (int(x + w), int(y + h)), color2, 2)
        #cv2.putText(col_images[i], "{:.2f}m2".format(area), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color3, 2)

    # add contours to original frames
    outputFrames.append(col_images[i])
print('contours are fully detected')
########################################################################################################################
# specify video name
pathOut = videoName + 'Contours.mp4' # or .mp4v
# specify frames per second
fps = 14.0
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, (videoWidth, videoHight))
for i in range(len(outputFrames)):
    # collecting frames in a single video
    out.write(outputFrames[i])
out.release()
print('video is released!')