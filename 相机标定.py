# 加载环境包
import cv2
import numpy as np
# from PIL import  Image

# 图片简单处理


img = cv2.imread('1.jpg')  # 读取图片

GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化

GrayImage = cv2.medianBlur(GrayImage, 25)  # 中值模糊，medianBlur()函数使用中值滤波器来平滑图像。

# 阈值处理，输入图片默认为单通道灰度图片

ret, th1 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_TOZERO)  # 固定阈值二值化

# threshold为固定阈值二值化

# 第二参数为阈值

# 第三参数为当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值（一般情况下，都是256色，所以默认最大为255）

# thresh_binary是基于直方图的二值化操作类型，配合threshold一起使用。此外还有cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV；
# cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV

th2 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)

# adaptiveThreshold自适应阈值二值化，自适应阈值二值化函数根据图片一小块区域的值来计算对应区域的阈值，从而得到也许更为合适的图片。

# 第二参数为当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值（一般情况下，都是256色，所以默认最大为255）

# 第三参数为阈值计算方法，类型有cv2.ADAPTIVE_THRESH_MEAN_C，cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# 第四参数是基于直方图的二值化操作类型，配合threshold一起使用。此外还有cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV；
# cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV

# 第五参数是图片中分块的大小

# 第六参数是阈值计算方法中的常数项

th3 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)

# 同上

kernel = np.ones((6, 9), np.uint8)  # 创建全一矩阵，数值类型设置为uint8

erosion = cv2.erode(th2, kernel, iterations=1)  # 腐蚀处理


dilation = cv2.dilate(erosion, kernel, iterations=1)  # 膨胀处理

imgray = cv2.Canny(erosion, 30, 100)  # Canny算子边缘检测

# cv2.imshow("s", dilation)
circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1,20,param1=100,param2=20,minRadius=0,maxRadius=23) # 霍夫圆变换

# 第3参数默认为1

# 第4参数表示圆心与圆心之间的距离（太大的话，会很多圆被认为是一个圆）

# 第5参数默认为100

# 第6参数根据圆大小设置(圆越小设置越小，检测的圆越多，但检测大圆会有噪点)

# 第7圆最小半径

# 第8圆最大半径

circles = np.uint16(np.around(circles))

# np.uint16数组转换为整数，16位，0-65535

# np.around返回四舍五入后的值


P = circles[0]  # 去掉circles数组一层外括号

for i in P:
    # 画出外圆

    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 第二参数（）内是圆心坐标，第三参数是半径，第四参数（）内是颜色，第五参数是线条粗细

    # 画出圆心

    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

print("圆的个数是：")

print(len(P))

# 打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
fresult = open('result.txt', 'w')

for i in P:
    r = int(i[2])

    x = int(i[0])

    y = int(i[1])

    print("圆心坐标为：", (x, y))

    # 将圆心写入文件
    fresult.writelines("圆心坐标为：(" + str(i[0]) + "," + str(i[1]) + ")" + "\n")

    print("圆的半径是：", r)

cv2.imshow('detected circles', img)  # 第一参数为窗口名称

cv2.waitKey(0)  # 无穷大等待时间

cv2.destroyAllWindows()
