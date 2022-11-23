import cv2
import pyzbar.pyzbar as pyzbar

def Anynize(gray):
    #解析图片
    barcodes = pyzbar.decode(gray)

    for barcode in barcodes:
        #提取二维码信息，并画出边框
        (x,y,w,h) = barcode.rect
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),2)

        #转换编码格式，以字符串显示
        barcodeData = barcode.data.decode('utf-8')

        print("{}".format(barcodeData))

    return gray



def detect():
    cap = cv2.VideoCapture(1)
    while True:
        ret,img = cap.read()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        im = Anynize(gray)

        cv2.imshow("img", im)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()


# import qrcode
# qr=qrcode.QRCode(version = 2,error_correction = qrcode.constants.ERROR_CORRECT_L,box_size=10,border=10,)
# qr.add_data("时代内涵的教案大数据库不贵啊撒就是不管")
# qr.make(fit=True)
# img = qr.make_image()
# img.show()
# img.save('test.jpg')