import glob,os
from PIL import Image
# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sn

os.chdir(r'my_dir')

def CountLeafPixelsFromImage(imageName):
    img = cv.imread(imageName)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    sensitivity = 20
    hsv_color1 = np.asarray([35, 45,20])   
    hsv_color2 = np.asarray([90, 255,255]) 
    mask = cv.inRange(img_hsv, hsv_color1, hsv_color2)
    mask3 = np.zeros_like(img)
    mask3[:, :, 0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask
    numberOfPixels = np.sum(mask3 == 255)
    cv.imwrite(imageName+'.PNG', mask)
    return numberOfPixels

df = pd.DataFrame(columns=['count','identi','plotNo','ImageName'])

## doing all files

threeFolders = []### add three folders into the list that contain the images of an individual treatment. Make sure the order is the same as in the indentifyer list

identifyer = ['MS','poor','struv']

ident=-1

for folder in threeFolders:
    os.chdir(folder)
    ident+=1
    ListOfImages = []
    for file in glob.glob("*.JPG"):
        ListOfImages.append(file)

    countedPixels = []

    for i in ListOfImages:
        countedPixels.append(CountLeafPixelsFromImage(i))
        dftoADD = pd.DataFrame([[CountLeafPixelsFromImage(i),identifyer[ident],ident,str(i)]],columns=['count','identi','plotNo','ImageName'])
        df = pd.concat([df,dftoADD])
    sn.histplot(countedPixels,bins = 10)
