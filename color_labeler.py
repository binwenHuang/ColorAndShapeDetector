from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2


class ColorLabeler:
    def __init__(self):

        colors = OrderedDict({
            "red": (255, 166, 146),
            "green": (131, 224, 164),
            "blue": (0, 156, 255),
            'yellow':(255,255,255),
             'orange':(255,255,201)})


        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)


        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self,image,c):
        mask=np.zeros(image.shape[:2],dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        minDist = (np.inf, None)

        for (i, row) in enumerate(self.lab):

            d = dist.euclidean(row[0], mean)


            if d < minDist[0]:
                minDist = (d, i)

        return self.colorNames[minDist[1]]