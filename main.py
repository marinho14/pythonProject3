# This is a sample Python script.
import cv2
import numpy as np
import os
import sys
import metodos as mt


if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    N=2
    lista=mt.descom(image,N)
    imagen_ILL = lista[-1][-1]
    img_int=mt.Interpolacion(4,imagen_ILL)
    num = 1
    for a in (lista):
        for b in a:
            cv2.imshow("Ciclo 1" + str(num), b)
            num += 1
    cv2.imshow("Imagen Original",image)
    cv2.imshow("Imagen Interpolada", img_int)
    cv2.waitKey(0)



