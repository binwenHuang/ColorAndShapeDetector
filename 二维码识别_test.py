import cv2
import pyzbar.pyzbar as pyzbar


def anaize(gray):
    barcodes = pyzbar.decode(gray)

    for barcode in barcodes:
        (x,y,w,h) = barcode.rect
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,123,231),2)

        barcodeData = barcode.data.decode('utf-8')

        print('{}'.format(barcodeData))

    return gray



def detect():
    cap = cv2.VideoCapture(1)
    while True:
        ret,img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = anaize(gray)

        cv2.imshow('img',img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()