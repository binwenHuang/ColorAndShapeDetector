import cv2


def detect(c):
        shape='unidentified'
        perimeter=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.04*perimeter,True)# contain list of vertices "kodkodim"

        if len(approx)==3:
            shape='triangle'
        elif len(approx)==4:

            (x,y,w,h)=cv2.boundingRect(approx)
            aspect_ratio=w/float(h)

            if aspect_ratio>=0.95 or aspect_ratio <=1.05:
                shape='square'
            else:
                shape='rectangle'
        elif len(approx)==5:
            shape='pentagon'
        else:
            shape='circle'
        return shape
