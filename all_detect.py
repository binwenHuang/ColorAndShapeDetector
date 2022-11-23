import imutils
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2


class ColorLabeler:
    def __init__(self):
        # 定义颜色
        colors = OrderedDict({
            "red": (255, 166, 146),
            "green": (131, 224, 164),
            "blue": (0, 156, 255),
            'yellow':(255,255,255),
             'orange':(255,255,201)})

        #定义标签属性并创建0矩阵
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")

        # 将颜色与标签进行拆分
        self.colorNames = []
        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)

        #转灰度
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)


    def label(self,image,c):
        #设置mask0矩阵
        mask=np.zeros(image.shape[:2],dtype='uint8')

        #画边框
        cv2.drawContours(mask, [c], -1, 255, -1)

        # 求平均
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        # 循环读取
        minDist = (np.inf, None)
        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
                minDist = (d, i)

        return self.colorNames[minDist[1]]



def detect(c):
    #初始化形状
    shape = 'unidentified'

    #周长近似值
    perimeter = cv2.arcLength(c, True)

    #边数近似值
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)  # contain list of vertices "kodkodim"

    #根据边数判断形状
    if len(approx) == 3:
        shape = 'triangle'

    elif len(approx) == 4:
        #取出图形坐标与宽高
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        #判断正方形与长方形
        if aspect_ratio >= 0.95 or aspect_ratio <= 1.05:
            shape = 'square'
        else:
            shape = 'rectangle'
    elif len(approx) == 5:
        shape = 'pentagon'
    else:
        shape = 'circle'
    return shape



cap = cv2.VideoCapture(1)
while True:
    _, image = cap.read()

    #重置大小与比例
    resized=imutils.resize(image,width=300)
    ratio=image.shape[0]/float(resized.shape[0]) # ratio of resizing

    #进行高斯模糊处理
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    #转灰度
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    cv2.imshow('gary', gray)

    #对灰度图像进行二值化处理
    thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('img',thresh)

    #寻找图像中的轮廓
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #实例化ColorLabeler这个类
    color_d=ColorLabeler()

    #遍历找到的轮廓
    for c in cnts:
        M = cv2.moments(c)
        try:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        except:
            cX = 0
            cY = 0

        #调用detect检测形状,label检测颜色
        shape = detect(c)
        color=color_d.label(lab,c)

        #形状参数
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")

        #画出形状
        cv2.drawContours(image, [c], -1, (160, 38, 191), 2)

        cv2.putText(image, f'{color}, {shape}', (cX-20, cY),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, (160, 38, 191), 2)

        cv2.imshow("Image", image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



