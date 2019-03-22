# coding: utf-8
import cv2
import numpy as np
import sys

colors_thre = {"green": [(42, 70, 100), (70, 255, 255)], "blue": [(95, 100, 100), (120, 255, 255)], "red": [(0, 100, 100), (10, 255, 255)]}
img = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
img2 = img.copy()

blur = cv2.GaussianBlur(img, (7, 7), 0)

hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, colors_thre[sys.argv[1]][0], colors_thre[sys.argv[1]][1])

res = cv2.bitwise_and(img2, img2, mask=mask)

res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

kernel = np.ones((10,10),np.uint8)
closed_gray = cv2.morphologyEx(res_gray, cv2.MORPH_CLOSE, kernel)

contours, h = cv2.findContours(closed_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_with_contours = img2.copy()
cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)

filtered_contours = []

for c in contours:
    M = cv2.moments(c)
    if M['m00'] < 2000:
        continue
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if float(M['m00'])/hull_area < 0.9:
        continue
    filtered_contours.append(c)

img_selected_contours = img2.copy()

cv2.drawContours(img_selected_contours, filtered_contours, -1, (255, 0, 0), 3)

puck_poses = []
for c in filtered_contours:
  M = cv2.moments(c)
  puck_poses.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))

img_poses = img2.copy()
for pt in puck_poses:
  cv2.line(img_poses, (pt[0]-5, pt[1]), (pt[0]+5, pt[1]), (0, 255, 0))
  cv2.line(img_poses, (pt[0], pt[1]-5), (pt[0], pt[1]+5), (0,255,0))

cv2.imshow("plop", img2)
cv2.imshow("plip", mask)
cv2.imshow("plup", res)
cv2.imshow("closed gray", closed_gray)
cv2.imshow("plyp", img_with_contours)
cv2.imshow("filtered", img_selected_contours)
cv2.imshow("poses", img_poses)
cv2.waitKey(0)
cv2.destroyAllWindows()
