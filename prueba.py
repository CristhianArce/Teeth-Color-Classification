import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
import os
from xml.dom import minidom

path = 'C:\\Users\\Cristhian\\Documents\\TEETH-RECOGNITION-WITH-MACHINE-LEARNING\\DATASET\\'
path_label = 'C:\\Users\\Cristhian\\Documents\\TEETH-RECOGNITION-WITH-MACHINE-LEARNING\\Labels\\XMLTags\\'


def getListaArchivos(path):
    files = os.listdir(path)
    return files


def getXMLCoordinates(file_name):
    doc = minidom.parse(path_label+file_name + ".xml")
    '''width = doc.getElementsByTagName("width")[0].firstChild.data
    height = doc.getElementsByTagName("height")[0].firstChild.data'''
    xmin = int(doc.getElementsByTagName("xmin")[0].firstChild.data)
    xmax = int(doc.getElementsByTagName("xmax")[0].firstChild.data)
    ymin = int(doc.getElementsByTagName("ymin")[0].firstChild.data)
    ymax = int(doc.getElementsByTagName("ymax")[0].firstChild.data)
    return xmax, xmin, ymax, ymin


def recorrerCoordenadas(image, xmax, xmin, ymax, ymin):
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            print("I: "+str(i)+" J:"+str(j))


def recortarImagen(image, xmax, xmin, ymax, ymin):
    im = image[ymin:ymax, xmin:xmax]
    return im


def bgr_to_hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, None, fx=1/3, fy=1/3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv


def smile_segmentation(image):
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    min_red = np.array([10, 0, 0])
    max_red = np.array([120, 256, 256])
    image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    # show_mask(image_red1)
    # overlay_mask(image_red1, image)
    return image_red1


def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    show(img)


def show(image):
    # Figure size in inches
    plt.figure(figsize=(15, 15))
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')


def applyMorphologicFilter(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # Fill small gaps
    image_red_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    show_mask(image_red_closed)
    # Remove specks
    image_red_closed_then_opened = cv2.morphologyEx(image_red_closed, cv2.MORPH_OPEN, kernel)
    show_mask(image_red_closed_then_opened)
    return image_red_closed_then_opened


def find_biggest_contour(image):
    # Copy to prevent modification
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Isolate largest contour
    biggest_contour = max(contours, key=cv2.contourArea)
    # Draw just largest contour
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def leerImages():
    files = getListaArchivos(path)
    for file in files:
        im = plt.imread(path + file)
        file_name = file[:-4]
        xmax, xmin, ymax, ymin = getXMLCoordinates(file_name)
        ima = recortarImagen(im, xmax, xmin, ymax, ymin)
        #hsv = bgr_to_hsv(ima)
        mask = smile_segmentation(ima)
        morph_filter = applyMorphologicFilter(mask)
        big_contour, mask = find_biggest_contour(morph_filter)
        overlay_mask(mask, ima)
        # plt.imshow(hsv)
        plt.show()

        # print(hsv)
        #print("xmax: " + str(xmax) + " xmin: " + str(xmin) + " ymin: " + str(ymin) + " ymax: " + str(ymax) )
        #print(im[ymin:ymax ,xmin:xmax])
        '''plt.scatter(x=[xmax, xmin], y=[ymax, ymin], c='r', s=40)
        plt.scatter(x=[xmin, xmax], y=[ymax, ymin], c='g', s=40)
        '''


leerImages()
