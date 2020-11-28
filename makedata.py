import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video = cv2.VideoCapture('ring-vids/three.mp4')

# Check if camera opened successfully
if (video.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
numFrame=0
while(video.isOpened()):
  # Capture frame-by-frame

  numFrame+=1

  ret, frame = video.read()

  if ret == True:

    # Display the resulting frame
    #cv2.imshow('Frame',frame)
    if numFrame%2==0:
        cv2.imwrite("three" + str(numFrame) + ".png",frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
video.release()

# Closes all the frames
cv2.destroyAllWindows()
