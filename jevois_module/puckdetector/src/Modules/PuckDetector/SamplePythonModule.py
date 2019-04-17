import libjevois as jevois
import cv2
import numpy as np


## Detects pucks for eurobot 2019
#
# Add some description of your module here.
#
# @author The Rusty Ducks
# 
# @videomapping YUYV 640 480 30 YUYV 640 480 30 TheRustyDucks PuckDetector
# @email therustyducks@gmail.com
# @address 123 first street, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by The Rusty Ducks
# @mainurl 
# @supporturl 
# @otherurl 
# @license 
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PuckDetector:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        self.colors_thre = {"green": [(42, 70, 100), (70, 255, 255)], "blue": [(95, 100, 100), (120, 255, 255)],
                            "red": [(175, 100, 100), (180, 255, 255), (0, 100, 100), (10, 255, 255)]}

    def processNoUSB(self, inframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR. If you need a
        # grayscale image, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB() and getCvRGBA():
        inimg = inframe.getCvBGR()

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        pucks_color = self.find_pucks(inimg)

        jevois.sendSerial("{}:{}:{}".format(self.serialize_puck_list(pucks_color["red"]),
                                            self.serialize_puck_list(pucks_color["green"]),
                                            self.serialize_puck_list(pucks_color["blue"])))

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR. If you need a
        # grayscale image, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB() and getCvRGBA():
        inimg = inframe.getCvBGR()

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        pucks_color, outimg = self.find_pucks(inimg, with_output=True)

        jevois.sendSerial("{}:{}:{}".format(self.serialize_puck_list(pucks_color["red"]),
                                            self.serialize_puck_list(pucks_color["green"]),
                                            self.serialize_puck_list(pucks_color["blue"])))
        # Write a title:
        cv2.putText(outimg, "JeVois PuckDetector", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Convert our output image to video output format and send to host over USB:
        outframe.sendCv(outimg)

    @staticmethod
    def serialize_puck_list(l):
        s = ""
        for pt in l:
            s += str(pt[0]) + "," + str(pt[1]) + ";"
        return s[:-1]

    def find_pucks(self, inimg, with_output=False):
        blur = cv2.GaussianBlur(inimg, (7, 7), 0)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        if with_output:
            output_img = inimg.copy()

        pucks_color = {}

        for d_color in self.colors_thre.keys():
            mask = cv2.inRange(hsv, self.colors_thre[d_color][0], self.colors_thre[d_color][1])
            if d_color == "red":
                # Red is special since it wraps around H:0
                mask += cv2.inRange(hsv, self.colors_thre[d_color][2], self.colors_thre[d_color][3])

            res = cv2.bitwise_and(inimg, inimg, mask=mask)

            res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((10, 10), np.uint8)
            closed_gray = cv2.morphologyEx(res_gray, cv2.MORPH_CLOSE, kernel)

            contours, h = cv2.findContours(closed_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = []

            for c in contours:
                M = cv2.moments(c)
                if M['m00'] < 2000:
                    continue
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if float(M['m00']) / hull_area < 0.9:
                    continue
                filtered_contours.append(c)

            puck_poses = []
            for c in filtered_contours:
                M = cv2.moments(c)
                puck_poses.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))

            if with_output:
                for pt in puck_poses:
                    cv2.line(output_img, (pt[0] - 5, pt[1]), (pt[0] + 5, pt[1]), (0, 255, 255))
                    cv2.line(output_img, (pt[0], pt[1] - 5), (pt[0], pt[1] + 5), (255, 255, 0))

            pucks_color[d_color] = []
            for c in filtered_contours:
                x, y, w, h = cv2.boundingRect(c)
                cropped_hsv = hsv[y:y + h, x:x + w].copy()
                cropped_mask = 255 - cv2.inRange(cropped_hsv, (0, 0, 0), (255, 255, 90))
                kernel = np.ones((12, 12), np.uint8)
                cropped_closed = cv2.dilate(cropped_mask, kernel, iterations=1)
                contours, h = cv2.findContours(cropped_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                center_contour = None
                center_contour_m = {'m00': 0}
                for i, ci in enumerate(contours):
                    if h[0][i][2] != -1:
                        continue  # Looking for last in trees (should be looking for max depth...)
                    M = cv2.moments(ci)
                    if M['m00'] > center_contour_m['m00']:
                        center_contour = ci
                        center_contour_m = M
                center_pose = (x + int(center_contour_m['m10'] / center_contour_m['m00']),
                               y + int(center_contour_m['m01'] / center_contour_m['m00']))
                pucks_color[d_color].append(center_pose)

                if with_output:
                    color = cv2.cvtColor(np.uint8([[self.colors_thre[d_color][0]]]), cv2.COLOR_HSV2BGR)
                    cv2.line(output_img, (center_pose[0] - 5, center_pose[1]), (center_pose[0] + 5, center_pose[1]),
                             color)
                    cv2.line(output_img, (center_pose[0], center_pose[1] - 5), (center_pose[0], center_pose[1] + 5),
                             color)
        if with_output:
            return pucks_color, output_img
        else:
            return pucks_color
