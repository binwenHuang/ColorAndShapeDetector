import imutils
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class ColorLabel:
    def __init__(self):
        color = OrderedDict({
            "red": (255, 166, 146),
            "green": (131, 224, 164),
            "blue": (0, 156, 255),
            'yellow': (255, 255, 255),
            'orange': (255, 255, 201)})

        self.lab = np.zeros((len(color),1,3),dtype='uint8')

        self.colorName = []
        for (i,(name,rgb)) in enumerate(color.items()):
            self.lab[i] = rgb
            self.colorName.append(name)

        self.lab = cv2.cvtColor(self.lab,cv2.COLOR_BGR2GRAY)

    def label(self,image,c):
        mask = np.zeros(image.shape[:2],dtype='uint8')
        cv2.drawContours(mask,[c],-1,255,-1)
        mask = cv2.erode(mask,None,iterations=2)
        mean = cv2.mean(image,mask = mask)[:3]
        minDist = (np.inf,None)
        for (i,row) in enumerate(self.lab):
            d = dist.euclidean(row[0],mean)
            if d < minDist[0]:
                minDist = (d,i)

        return self.colorName[minDist[1]]


def detect(c):
    shape = 'unidentified'
    perimeter = cv2.arcLength(c,True)

    approx = cv2.approxPolyDP(c,0.04*perimeter,True)

    if len(approx) == 3:
        shape = 'trigle'
    elif len(approx) == 4:
        (x,y,w,h) = cv2.boundingRect(approx)
        asper = w/float(h)
        if asper >= 0.95 or asper <= 1.05:
            shape = 'squre'
        else:
            shape = 'rectangle'
    elif len(approx) == 5:
        shape = ''

    else:
        shape = 'circle'
    return shape


cap = cv2.VideoCapture(1)
while True:
    ret,image = cap.read()

    #chongzhi
    resized = imutils.resize(image,width=300)
    ratio = image.shape[0]/float(resized.shape[0])

    #gauss
    gauss = cv2.GaussianBlur(resized,(5,5),0)

    #gray
    gray = cv2.cvtColor(gauss,cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

    #thresh
    thresh = cv2.threshold(gray,115,255,cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh',thresh)

    #cnts
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    color_b = ColorLabel()

    #遍历轮廓
    for c in cnts:
        M = cv2.moments(c)

        try:
            cX = int((M['m10']/M['m00'])*ratio)
            cY = int((M['m01'] / M['m00']) * ratio)
        except:
            cX= 0
            cY = 0

        shape = detect(c)
        color = color_b.label(lab,c)

        # 形状参数
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")

        cv2.drawContours(image,[c],-1,(123,123,123),2)
        cv2.putText(image,f'{color},{shape}',(cX-20,cY),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(123,123,123),2)

        cv2.imshow('image',image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()









