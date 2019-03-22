# coding: utf-8
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
img2 = img.copy()

img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img_canny = cv2.Canny(img_gray, 200, 20)

circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 60, param1=200, param2=20, minRadius=0, maxRadius=0)

img_circles = img2.copy()


for c in circles[0]:
  print(c)
  pt = (int(c[0]), int(c[1]))
  radius = int(c[2])
  cv2.circle(img_circles, pt, 3, (0, 255, 255), -1)
  cv2.circle(img_circles, pt, radius, (0, 0, 255), 1)


cv2.imshow("plop", img2)
cv2.imshow("canny", img_canny)
cv2.imshow("circles", img_circles)
cv2.waitKey(0)
cv2.destroyAllWindows()
