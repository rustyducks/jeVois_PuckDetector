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


for c in filtered_contours:
  x, y, w, h = cv2.boundingRect(c)
  cropped_hsv = hsv[y:y+h, x:x+w].copy()
  cv2.imshow("cropped", cropped_hsv)
  cropped_mask = 255 - cv2.inRange(cropped_hsv, (0, 0, 0), (255, 255, 90))
  cv2.imshow("cropped_mask", cropped_mask)
  kernel = np.ones((12,12),np.uint8)
  cropped_closed = cv2.dilate(cropped_mask, kernel, iterations=1)
  cv2.imshow("cropped_closed", cropped_closed)
  cropped_contour = cropped_hsv.copy()
  contours, h = cv2.findContours(cropped_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  center_contour = None
  center_contour_m = {'m00':0}
  for i, ci in enumerate(contours):
    if h[0][i][2] != -1:
      continue  # Looking for last in trees (should be looking for max depth...)
    M = cv2.moments(ci)
    if M['m00'] > center_contour_m['m00']:
      center_contour = ci
      center_contour_m = M
  cv2.drawContours(cropped_contour, [center_contour], -1, (255, 0, 0), 3)
  cv2.imshow("inside contour", cropped_contour)
  center_pose = (x + int(center_contour_m['m10']/center_contour_m['m00']), y + int(center_contour_m['m01']/center_contour_m['m00']))
  cv2.line(img_poses, (center_pose[0]-5, center_pose[1]), (center_pose[0]+5, center_pose[1]), (0, 0, 255))
  cv2.line(img_poses, (center_pose[0], center_pose[1]-5), (center_pose[0], center_pose[1]+5), (0,0,255))
  

cv2.imshow("plop", img2)
cv2.imshow("plip", mask)
cv2.imshow("plup", res)
cv2.imshow("closed gray", closed_gray)
cv2.imshow("plyp", img_with_contours)
cv2.imshow("filtered", img_selected_contours)
cv2.imshow("poses", img_poses)
cv2.waitKey(0)
cv2.destroyAllWindows()
