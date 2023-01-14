import os
import cv2
import argparse
from PIL import Image
import numpy as np
from sklearn.metrics import *
from models.rscdnet import modelDef

gtappend   = []
predappend = []
erodappend = []

def testing(testPath, imageA_folder, imageB_folder, mask_folder, image_size, directory, modelName):
    model = modelDef(image_size)
    model.load_weights(os.path.join(directory, modelName + '.h5'))

    images = os.listdir(os.path.join(testPath, imageA_folder))
    
    for im in images:
        x1  = Image.open(os.path.join(testPath, imageA_folder, im))
        x2  = Image.open(os.path.join(testPath, imageB_folder, im))
        gt = cv2.imread(os.path.join(testPath, imageA_folder, im),0)
        a=np.reshape(np.array(x1),(1, image_size[0], image_size[1], image_size[2]))/255.
        b=np.reshape(np.array(x2),(1, image_size[0], image_size[1], image_size[2]))/255.
        y = model.predict([a,b])
        y=np.reshape(y,(image_size[0], image_size[1], 2))
        result = np.argmax(y,axis = 2)
        result = result*255.
        gtappend.append(gt)
        predappend.append(result)

    g = np.stack(gtappend, axis=0)
    p = np.stack(predappend, axis=0)


    gt=(g.ravel()/255).astype('int')
    pd=(p.ravel()/255).astype('int')
    f1 = f1_score(gt,pd)
    print("F1 SCORE:", f1)
    kappa = cohen_kappa_score(gt,pd)
    print("Kappa:",kappa)
    accuracy = accuracy_score(gt,pd) 
    print("Accuracy:",accuracy)
    jaccard = jaccard_score(gt,pd)
    print("Jaccard Score:",jaccard)
    precision = precision_score(gt,pd) 
    print("Precision:",precision)
    recall = recall_score(gt,pd)
    print(np.unique(gt),np.unique(pd))
    print("Recall:",recall)

    cf_matrix = confusion_matrix(gt,pd)
    cf_matrix/cf_matrix.astype(np.float).sum(axis=1)


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--testPath','-t', type=str, help='Path to train folder')
        parser.add_argument('--imageA_folder', '-a', type=str,  default='img1', help='Image A folder')
        parser.add_argument('--imageB_folder', '-b', type=str, default='img2', help='Image B folder')
        parser.add_argument('--mask_folder', '-m', type=str, default='mask', help='Mask folder')
        parser.add_argument('--image_size', '-i', type=int, default=1024, help='size of the final/resized image from train generator')
        parser.add_argument('--directory', '-d', type =str, default='models', help='Directory of the trained model ')
        parser.add_argument('--modelName', '-mn', type=str, default='proposed', help=' name of the model, modelName.h5 ')
        
        FLAGS = parser.parse_args()
        FLAGS.image_size = (FLAGS.image_size, FLAGS.image_size, 3)
        
        testing(FLAGS.trainPath, FLAGS.imageA_folder, FLAGS.imageB_folder, FLAGS.mask_folder, FLAGS.image_size,
            FLAGS.directory, FLAGS.modelName)
    