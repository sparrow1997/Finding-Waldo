
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/WaldoBeach.jpg')
template = cv2.imread('images/waldo.jpg',0)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

h,w= template.shape[0],template.shape[1]
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = gray.copy()
    method = eval(meth)
    result = cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w,top_left[1] + h)
    cv2.rectangle(img,top_left,bottom_right,(255,255,0),2)
    cv2.imshow(meth,img)
    cv2.waitKey(0)
   
cv2.destroyAllWindows()

#Detailed Description



