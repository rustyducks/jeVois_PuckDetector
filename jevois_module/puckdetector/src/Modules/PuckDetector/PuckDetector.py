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
        self.colors_thre = {"green": [(42, 60, 80), (90, 255, 255)], "blue": [(95, 100, 80), (120, 255, 255)],
                            "red": [(140, 100, 70), (180, 255, 255), (0, 100, 70), (10, 255, 255)]}
        self.color_channel = {"blue": 0, "green":1, "red": 2}
        self.mean_theshold = {"blue": 0.4, "green":0.4, "red": 0.4}
        self.color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
        self.places_offset = [(0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)]

    def processNoUSB(self, inframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR. If you need a
        # grayscale image, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB() and getCvRGBA():
        inimg = inframe.getCvBGR()

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        pucks_circles = self.find_pucks(inimg)
        jevois.LINFO("FUUUUCK")
        jevois.sendSerial("{}".format(self.serialize_puck_list(pucks_circles)))

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR. If you need a
        # grayscale image, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB() and getCvRGBA():
        inimg = inframe.getCvBGR()

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        pucks_circles, outimg = self.find_pucks(inimg, with_output=True)

        jevois.sendSerial("{}".format(self.serialize_puck_list(pucks_circles)))
        
        # Write a title:
        cv2.putText(outimg, "JeVois PuckDetector", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        for i, data in enumerate(pucks_circles):
            x,y,r,avr, col = data
            cv2.putText(outimg, "{}, {}, {}, {}, {}".format(x,y,r,avr, col), (3, 40 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        


        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Convert our output image to video output format and send to host over USB:
        outframe.sendCv(outimg)

    @staticmethod
    def serialize_puck_list(pucks):
        s = ";".join(["{} {} {} {} {}".format(x,y,r,avr,col) for x,y,r,avr,col in pucks])
        return s
    
    def find_pucks(self, inimg, with_output=False):
        #inimg = self.trans(inimg)
        bordersize = 120#0
        img=cv2.copyMakeBorder(inimg, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
        h,w,ch=img.shape
        out = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cannyHighThresh = 80
        cannyLowThresh = 20
        canny = cv2.Canny(gray, cannyHighThresh, cannyLowThresh)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=250, param1=cannyHighThresh, param2=cannyLowThresh, minRadius=150, maxRadius=180)
        #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=100, param1=cannyHighThresh, param2=cannyLowThresh, minRadius=50, maxRadius=120)
        pucks = []
        outImg = np.zeros((h,w, 3), np.uint8)
        outImg[0:h, 0:w] = img[0:h, 0:w]
        if circles is not None:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #convert image to HSV space for color filtering
            #img_hsv = cv2.medianBlur(img_hsv, 5)
            circles = np.round(circles[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles):
                mask = np.zeros_like(gray)          #will be the mask that define the atoms
                mask_of_mask = np.zeros_like(gray)  #will be used to mask the borders on the previous mask
                cv2.rectangle(mask_of_mask, (bordersize, bordersize), (h+bordersize, w+bordersize), (255), -1)  #mask that exclude the borders
                cv2.circle(mask, (x,y), r, (255,255,255), -1)  #draw the detected circle on the mask
                mask = cv2.bitwise_and(mask, mask, mask=mask_of_mask) #exclude the (black) borders from the mask
                
                #jevois.LINFO("")
                for d_color in self.colors_thre.keys():
                    #filter image on color
                    #img_hsv = cv2.blur(img_hsv, (5,5)) #cv2.GaussianBlur(img_hsv, (11, 11), 5, 5)
                    
                    img_filtered = cv2.inRange(img_hsv, self.colors_thre[d_color][0], self.colors_thre[d_color][1])
                    if d_color == "red":
                        img_filtered += cv2.inRange(img_hsv, self.colors_thre[d_color][2], self.colors_thre[d_color][3])
                    #compute the mean of the filtered image, masked by the circle mask.
                    #m is the percentage of the detected circle that is of the color d_color
                    m=cv2.mean(img_filtered, mask=mask)[0]/255
                    #jevois.LINFO(d_color)
                    #jevois.LINFO(str(m))
                    if m > self.mean_theshold[d_color]:
                        pucks.append((x-bordersize,y-bordersize,r,m,d_color))
                        cv2.circle(outImg, (x, y), r, self.color[d_color], 4)
                        #debug >>>>>>>>>>>>>>>>>>>>
                        #if i < 3:
                        #     imm = cv2.bitwise_and(img_filtered, img_filtered, mask=mask)
                        #     coco = np.zeros_like(img_hsv)
                        #     coco[:,:,self.color_channel[d_color]] = imm
                        #     mulH, mulW, offH, offW = self.places_offset[i]
                        #     outImg[mulH*h:mulH*h+h, mulW*w:mulW*w+w] = coco
                        #debug <<<<<<<<<<<<<<<
                        break
                else:
                    #no color where m>mean_theshold, i.e, not the good color
                    cv2.circle(outImg, (x, y), r, (255, 255, 255), 4)
                    pass

        if with_output:
            return pucks, outImg
        else:
            return pucks
                
    def trans(self, image):
        maxWidth = 200
        maxHeight = 200
        dst = np.array([
        [0, 0],
        [640, 0],
        [640, 480],
        [0, 480]], dtype = "float32")
        
        #src = np.array([(139, 51), (526, 50), (594, 266), (78, 266)], dtype = "float32")
        src = np.array([(0, 0), (640, 0), (940, 480), (-300, 480)], dtype = "float32")
        M = cv2.getPerspectiveTransform(src, dst)
        jevois.LINFO(str(M))
        warped = cv2.warpPerspective(image, M, (640, 480))
        return warped
        
    
    
