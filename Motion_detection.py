# -*- coding: utf-8 -*-
import cv2
import os
import datetime
import numpy as np

def flame_sub(im1,im2,im3,th,blur):
    d1 = cv2.absdiff(im3, im2)
    d2 = cv2.absdiff(im2, im1)
    diff = cv2.bitwise_and(d1, d2)
    # True if the difference is smaller than the threshold.
    mask = diff < th
    # Generate an array with the same size as the background image
    im_mask = np.empty((im1.shape[0],im1.shape[1]),np.uint8)
    im_mask[:][:]=255
    # True parts (background) are painted in black.
    im_mask[mask]=0
    # Small noise reduction
    im_mask = cv2.medianBlur(im_mask,blur)

    return  im_mask

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    #cam.set(3, 640)  # Width
    #cam.set(4, 380)  # Heigh
    im1 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    im2 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    im3 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

    new_dir_path = 'data'
    save_flg = 0

    #Create a saving folder
    try:
        os.makedirs(new_dir_path)
    except FileExistsError:
        pass

    while True:
        # Difference between frames
        im_fs = flame_sub(im1,im2,im3,5,7)
        cv2.imshow("Motion Mask",im_fs)

        #Detecting contours
        cnts = cv2.findContours(im_fs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        ret, frame = cam.read()

        backup_frame = frame.copy()

        #Surround the outline with squares.
        for c in cnts:
           x,y,w,h = cv2.boundingRect(c)
           if w < 40: continue
           save_flg = 1
           cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 3)

        #Extracting contours
        # cv2.drawContours(frame,cnts,-1,(0,255,0),3)

        # motion detection
        if save_flg == 1:
            cv2.imwrite('data/' + str(datetime.datetime.now()) + '.jpg', backup_frame,[cv2.IMWRITE_JPEG_QUALITY,30])
            save_flg = 0

        cv2.imshow("Input",frame)


        im1 = im2
        im2 = im3
        im3 = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        key = cv2.waitKey(10)
        # Press the Esc key and you're done.
        if key == 27:
            cv2.destroyAllWindows()
            break
