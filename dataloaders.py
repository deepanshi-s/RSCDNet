import os 
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

color_dict = {0: (0),
              1: (255),
              }


def rgb_to_onehot(rgb_arr, color_dict):
    """ Converts the RGB arrays to one Hot arrays

    Args:
        rgb_arr (numpy.array): RGB image array
        color_dict (python dict): RGB to integer mapping 

    Returns:
        numpy.array: One hot encoded image array
    """
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.float32 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,1) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


def adjustData(img1, img2, mask):
    """Preprocess the data 

    Args:
        img1 (numpy.array): Image A
        img2 (numpy.array): Image B
        mask (numpy.array): mask 
    """
    onehot=rgb_to_onehot(mask[0],color_dict)
    onehot=np.reshape(onehot,((1,)+onehot.shape))
    img1 = img1/255.
    img2 = img2/255.
    return(img1, img2, onehot)
  


#Dataloader
def dataGenerator(aug_dict, train_path, image1_folder, image2_folder, mask_folder, batch_size, save_to_dir, image_size, seed = 1, 
                   image1_save_prefix = "image1", image2_save_prefix = "image2", mask_save_prefix = "mask"):
    """ Train Data loader

    Args:
        aug_dict : Dictionary to apply augmentations to the loaded images 
        train_path (string): Path to the root directory (train)
        image1_folder (string): Name of the image A folder 
        image2_folder (string): Name of the image B folder
        mask_folder (string): Name of the mask image folder
        batch_size (int): Batch size for training
        save_to_dir (str): Directory to save the augmented images
        image_size (tuple(int, int)): Dimension to resize the image to
        seed (int, optional): Seed used for shuffling. Defaults to 1.
        image1_save_prefix (str, optional): Defaults to "image1".
        image2_save_prefix (str, optional): Defaults to "image2".
        mask_save_prefix (str, optional): Defaults to "mask".

    Yields:
        Directory Iterator yielding tuples of(x1, x2, y) 
    """
    image1_datagen = ImageDataGenerator(**aug_dict)
    image2_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image1_generator = image1_datagen.flow_from_directory(
        directory = os.path.join(train_path, image1_folder),
        target_size = image_size,
        color_mode = 'rgb',
        class_mode = None,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image1_save_prefix,
        seed = seed
    )
    image2_generator = image2_datagen.flow_from_directory(
        directory = os.path.join(train_path, image2_folder),
        target_size = image_size,
        color_mode = 'rgb',
        class_mode = None,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image2_save_prefix,
        seed = seed
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory = os.path.join(train_path, mask_folder),
        target_size = image_size,
        color_mode = 'grayscale',
        class_mode = None,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = mask_save_prefix,
        seed = seed
    )
    train_generator = zip(image1_generator, image2_generator, mask_generator)
    for (img1, img2, mask) in train_generator:
        img1, img2, mask = adjustData(img1, img2, mask)
        yield [img1, img2], mask
        

def dataLoaders(trainPath, valPath, imageA_folder, imageB_folder, mask_folder, batch_size, image_size):
    """ DataLoader function for train and val

    Args:
        trainPath (str): Path to train folder
        valPath (str): Path to val folder
        imageA_folder (str): Image A folder 
        imageB_folder (str): Image B folder
        mask_folder (str): Label folder
        batch_size (int): Batch size 
        image_size (tuple(int, int)): Dimension to resize the image to

    Returns:
        _type_: _description_
    """
    trainDataloader = dataGenerator(aug_dict = dict(), train_path=trainPath, image1_folder=imageA_folder, image2_folder=imageB_folder, mask_folder=mask_folder,
                                    batch_size=batch_size, save_to_dir=None, image_size=image_size)
    
    valDataLoader = dataGenerator(aug_dict = dict(), train_path=valPath, image1_folder=imageA_folder, image2_folder=imageB_folder, mask_folder=mask_folder,
                                    batch_size=batch_size, save_to_dir=None, image_size=image_size)

    return trainDataloader, valDataLoader