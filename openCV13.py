import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('Heineken.jpg',0)
#gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#img1 = np.float32(gray)

img2 = cv2.imread('thats_me.jpg',0)
#gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img2 = np.float32(gray)



orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:10], None, flags = 2)
#gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
#gray = np.float32(img3)

plt.imshow(img3)
plt.show()
