import numpy as np
import cv2

img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
#changing pixels
px = img[55,55]
print (px)

#region of image
img[50:100, 100:150] = [255,255,255]
watch_face = img[40:50, 100:150]
#crop image
img[0:10, 0:50] = watch_face
#print (roi)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()