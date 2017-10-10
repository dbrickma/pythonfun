import os
import cv2
from PIL import Image

r = [x for x in os.listdir() if x.endswith('.png')]

class uc_iter():
    def __init__(self,x):
        self.x = x
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            result = cv2.imwrite(str(self.x[self.index]) + '.jpg', cv2.cvtColor(cv2.imread(self.x[self.index]), cv2.COLOR_BGR2GRAY))
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

class uc_itera():
    def __init__(self,x):
        self.x = x
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            result = cv2.imread(self.x[self.index])
            gry = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
            nf = cv2.imwrite(str(self.x[self.index]) + 'a' + '.jpg',gry)
        except IndexError:
            raise StopIteration
        self.index += 1
        return nf

class uc_iterb():
    def __init__(self,x):
        self.x = x
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            result = cv2.imread(self.x[self.index])
            gry = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
            blr = cv2.blur(gry,(5,5))
            nf = cv2.imwrite(str(self.x[self.index]) + 'b' + '.jpg',blr)
        except IndexError:
            raise StopIteration
        self.index += 1
        return nf

    #other way
    def iter_pics(x):
        im = cv2.imread(x)
        gray_imageA = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return cv2.imwrite(str(x) + '.png', gray_imageA)

    list(map(lambda x: iter_pics(x), r))