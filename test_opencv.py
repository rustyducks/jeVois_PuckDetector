# coding: utf-8
import cv2
import numpy as np
img = cv2.imread("/home/gbuisan/Documents/eurobot_puck_images/my_photo-20.jpg", cv2.IMREAD_COLOR)
blurred_img = cv2.GaussianBlur(img, (41, 41), 0, 0)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByColor = True
params.filterByConvexity = True
params.minConvexity = 0.5
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)
kp = detector.detect(blurred_img)
img2 = img.copy()
for k in kp:
    cv2.drawMarker(img2, tuple(int(i) for i in k.pt), (255, 0, 0))

print(len(kp))
cv2.imshow("plop", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
