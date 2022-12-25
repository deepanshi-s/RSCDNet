
import os
import cv2 
import sys
from skimage.exposure import match_histograms


def claheapproach(img,cliplimit = 2.0,tilesize = (8,8)):
    clahe=cv2.createCLAHE(clipLimit= cliplimit, tileGridSize=tilesize)
    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l,a,b=cv2.split(lab)  # split on 3 different channels
    l2=clahe.apply(l)  # apply CLAHE to the L-channel
    lab=cv2.merge((l2,a,b))  # merge channels
    outputimage=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return outputimage


def preprocess(path, path2):
    subFolders = os.listdir(path)
    for folder in subFolders:
        os.mkdir(os.path.join(path2), folder)
        newPath = os.path.join(path, folder)
        imageFolders = os.listdir(newPath)
        folderA = os.path.join(newPath, imageFolders[0])
        folderB = os.path.join(newPath, imageFolders[1])
        folderLabel = os.path.join(newPath, imageFolders[2])
        os.mkdir(os.path.join(path2, folder, folderA))
        os.mkdir(os.path.join(path2, folder, folderB))
        os.mkdir(os.path.join(path2, folder, folderLabel))
        imageList = os.listdir(folderA)
        for img in imageList:
            imgA = os.path.join(folderA, img)
            imgB = os.path.join(folderB, img)
            label = os.path.join(folderLabel, img)
            imgA = cv2.imread(imgA)                      #reference for matching histograms
            imgB = cv2.imread(imgB)               
            label = cv2.imread(label)        
            imgBMatched = match_histograms(imgB, imgA, multichannel=True)
            
            imgA = claheapproach(imgA)
            imgB = claheapproach(imgBMatched)
            
            cv2.imwrite(os.path.join(path2, folder, folderA), imgA)
            cv2.imwrite(os.path.join(path2, folder, folderB), imgB)
            cv2.imwrite(os.path.join(path2, folder, folderLabel), label)
            
            
            
if __name__ == "__main__":
    
    #pass path to the root directory of the dataset
    '''
    The dataset structure:
    train
        - A
            - im1.png
            - im2.png
            ...
        - B
            - im1.png
            - im2.png
            ...
        - label
            - im1.png
            - im2.png
            ...
    val
         - A
            - im1.png
            - im2.png
            ...
        - B
            - im1.png
            - im2.png
            ...
        - label
            - im1.png
            - im2.png
            ...
    test
         - A
            - im1.png
            - im2.png
            ...
        - B
            - im1.png
            - im2.png
            ...
        - label
            - im1.png
            - im2.png
            ...
    '''
    if(len(sys.argv)!=2):
        print("Incorrect arguments, pass the path to the dataset, and the path to save the processed dataset")
        sys.exit
    
    preprocess(sys.argv[1], sys.argv[2])
    
    
    
    