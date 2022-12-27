import argparse
import os
import keras
import argparse
import numpy as np
from keras import backend as K
from keras.models import *
from keras.optimizers import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,Callback
from dataloaders import dataLoaders
from models.rscdnet import modelDef



def wcceloss(y_true, y_pred):
    """ Weighted categorical cross entropy

    Args:
        y_true (tensor): Label tensor (Batch size, H, W, 2)
        y_pred (tensor): Predicted tensor (Batch size, H, W, 2)
    """
    weights = np.array([0.5,2.5])
    weights = K.constant(weights)
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred) * weights
    loss = 1-K.sum(loss,axis= -1)


def train(trainPath, valPath, imageA_folder, imageB_folder, mask_folder, batch_size, image_size, epochs, directory, modelName, len_validation_data, len_train_data):
    """ Train function for RSCDNet

    Args:
        trainPath (str): Path to train folder 
        valPath (str): Path to val folder
        imageA_folder (str): Image A folder 
        imageB_folder (str): Image B folder
        mask_folder (str): Label folder
        batch_size (int): batch Size while training
        image_size (tuple): size of the final/resized image from train generator (H, W, C)
        epochs (int): Number of epochs
        directory (str): Directory to save the trained model 
        modelName (str): Save the model with modelName.h5 
        len_validation_data (int): Number of samples for validation
        len_train_data (int): Number of samples for training
    """
    trainDataloader, valDataLoader = dataLoaders(trainPath, valPath, imageA_folder, imageB_folder, mask_folder, 1, image_size[:2])
    
    model = modelDef(image_size)
    
    model.compile(optimizer = keras.optimizers.Adam(lr = 0.0001), loss = wcceloss, metrics = ['accuracy'])
    
    model_checkpoint = ModelCheckpoint(os.path.join(directory, modelName) + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
    model.fit_generator(trainDataloader, validation_data = valDataLoader, validation_steps = len_validation_data/batch_size,
                        steps_per_epoch = len_train_data/batch_size, epochs = epochs, callbacks=[model_checkpoint]
                        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainPath','-t', type=str, help='Path to train folder')
    parser.add_argument('--valPath', '-v', type=str, help='Path to val folder ')
    parser.add_argument('--imageA_folder', '-a', type=str,  default='img1', help='Image A folder')
    parser.add_argument('--imageB_folder', '-b', type=str, default='img2', help='Image B folder')
    parser.add_argument('--mask_folder', '-m', type=str, default='mask', help='Mask folder')
    parser.add_argument('--batch_size', '-bs', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--image_size', '-i', type=int, default=1024, help='size of the final/resized image from train generator')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--directory', '-d', type =str, default='models', help='Directory to save the trained model ')
    parser.add_argument('--modelName', '-mn', type=str, default='proposed', help=' Save the model with modelName.h5 ')
    parser.add_argument('--len_validation_data', type=int, help='Number of samples for validation')
    parser.add_argument('--len_train_data', '-lt', type=int, default=0.7, help='Number of samples for training')
    FLAGS = parser.parse_args()
    FLAGS.image_size = (FLAGS.image_size, FLAGS.image_size, 3)
    
    train(FLAGS.trainPath, FLAGS.valPath, FLAGS.imageA_folder, FLAGS.imageB_folder, FLAGS.mask_folder, FLAGS.batch_size, FLAGS.image_size,
          FLAGS.epochs, FLAGS.directory, FLAGS.modelName, FLAGS.len_validation_data, FLAGS.len_train_data)
    
   