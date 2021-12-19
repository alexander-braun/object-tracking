import cv2 as cv
from tracker import *

# create tracker obj
tracker = EuclideanDistTracker()

cap = cv.VideoCapture("./resources/highway.mp4")

# Object detection for stable cam
object_detector = cv.createBackgroundSubtractorMOG2(100, 50)

while True:
  success, img = cap.read()
  
  # extract retion of interest
  roi = img[340: 720, 500: 800]
  
  # Obj detection
  mask = object_detector.apply(roi)
  _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
  
  contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  bboxes = []
  for cnt in contours:
    # calc area and remove small contours
    area = cv.contourArea(cnt)
    if area > 400:
      #cv.drawContours(roi, cnt, -1, (0, 255, 0), 2)
      x,y,w,h = cv.boundingRect(cnt)
      bboxes.append([x, y, w, h])
      cv.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 255), 2)
    
  # object tracking
  boxes_ids = tracker.update(bboxes)
  for box_id in boxes_ids:
    x, y, w, h, id = box_id
    cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 255), 2)
  
  cv.imshow('Vid', img)
  key = cv.waitKey(30)
  if key == 27:
    break
  
cap.release()
cv.destroyAllWindows()
