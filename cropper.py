import os
import random
import cv2 as cv

# Give the offset a bias to be larger, as seen in image acquisition
high_offset = 0
low_offset = 0

cyclesdir = "C:/Users/shane t/Desktop/neat/testing/cycles/"
opendir= "C:/Users/shane t/Desktop/neat/testing/opengl/"
for subdir, dirs, files in os.walk(opendir):
    for directory in dirs:
        for file in os.listdir(opendir + directory):
            filename = os.fsdecode(file)

            # Load opengl (reference) image
            img = cv.imread(opendir + directory + "/" + filename)

            # Get binary representation
            imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(imgray, 100, 255, 0)
            # Find contours
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            bb = cv.boundingRect(contours[0])
            # Randomise the box with a bias towards larger box as observed with pi cropping
            x, y, w, h = bb
            '''[0] + random.randint(-high_offset, low_offset), bb[1] + random.randint(-high_offset, low_offset),\
                         bb[2] + random.randint(0, 2*high_offset), bb[3] + random.randint(0, 2*high_offset)'''

            # Draw rectangle and display for myself
            #cv.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            #cv.imshow('image', img)
            #cv.waitKey(0)

            # Open the properly rendered image
            img = cv.imread(cyclesdir + directory + "/" + filename)
            # Crop based on box generated on reference image
            cropped_img = img[y:y + h, x:x + w]

            # Show cropped image
            #cv.imshow('image', cropped_img)
            #cv.waitKey(0)

            # Save cropped image
            #print("C:/Users/shane t/Desktop/dataset/" + directory + "/" + filename)
            # Save image
            cv.imwrite("C:/Users/shane t/Desktop/test_images/" + directory + "/" + filename, cropped_img)

