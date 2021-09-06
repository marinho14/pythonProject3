# Se importan las librerias
import cv2
import numpy as np
import os
import sys
import metodos as mt ## Se exportan los metodos del archivo metodos.py


if __name__ == '__main__':
    path = sys.argv[1] ## Se define el path de la imagen
    image_name = sys.argv[2]  ## Se define el nombre de la imagen a procesar
    path_file = os.path.join(path, image_name) ## Se define la ruta completa la imagen
    image = cv2.imread(path_file) ## Se lee la imagen y se iguala
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ## Se convierte la imagen a grises
    N=2  ## Se define el N para la descomposicion
    lista=mt.descom(image,N)  ## Se usa el metodo descom y se crea una lista con las imagenes resultantes
    imagen_ILL = lista[-1][-1]  ## Se usa la ultima imagen de la lista (ILL)
    img_int=mt.Interpolacion(4,imagen_ILL)  ## Se interpola la ultima imagen de "lista"
    # Se usa un for para imprimir las imagenes obtenidas de la descomposicion realizada"
    num = 1
    for a in (lista):
        for b in a:
            cv2.imshow("Ciclo 1" + str(num), b)
            num += 1
    #Se imprimen las imagenes descompuestas, la original y la interpolada
    cv2.imshow("Imagen Original",image)
    cv2.imshow("Imagen Interpolada", img_int)
    cv2.waitKey(0)