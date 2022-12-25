from keras.models import *
from keras.layers import *

def modelDef(input_shape):
    input_size=(input_shape)
    input_1=Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #model_1 = Model(inputs = input_1, outputs = conv5)

    #VGG2

    input_2=Input(input_size)
    conv1_x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_2)
    conv1_x = Dropout(0.2)(conv1_x)
    conv1_x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_x)
    pool1_x = MaxPooling2D(pool_size=(2, 2))(conv1_x)
    conv2_x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_x)
    conv2_x = Dropout(0.2)(conv2_x)
    conv2_x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_x)
    pool2_x = MaxPooling2D(pool_size=(2, 2))(conv2_x)
    conv3_x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_x)
    conv3_x = Dropout(0.2)(conv3_x)
    conv3_x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_x)
    pool3_x = MaxPooling2D(pool_size=(2, 2))(conv3_x)
    conv4_x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_x)
    conv4_x = Dropout(0.2)(conv4_x)
    conv4_x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_x)
    pool4_x = MaxPooling2D(pool_size=(2, 2))(conv4_x)
    conv5_x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4_x)
    conv5_x = Dropout(0.2)(conv5_x)
    conv5_x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_x)
    
    CONV5 = Concatenate(name = 'concat')([conv5, conv5_x])

    #aspp
    #rate 6
    b1 = Conv2D(1024, 3, dilation_rate=(6, 6), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name ='b1' )(CONV5)
    b1 = Dropout(0.5)(b1)

    #rate 12
    b2 = Conv2D(1024, 3, dilation_rate=(12, 12), activation='relu', padding = 'same', kernel_initializer = 'he_normal' , name = 'b2')(CONV5)
    b2 = Dropout(0.5)(b2)

    #rate 18
    b3 = Conv2D(1024, 3, dilation_rate=(18, 18), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name = 'b3' )(CONV5)
    b3 = Dropout(0.5)(b3)

    #rate 24
    b4 = Conv2D(1024, 3, dilation_rate=(24, 24), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name = 'b4' )(CONV5)
    b4 = Dropout(0.5)(b4)

    b5 = Add(name = 'add')([b1, b2, b3, b4])

    up11 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(b5))
    up12 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up11)
    up12 = Dropout(0.2)(up12)
    up13 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up12)
    up21 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up13))
    up22 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up21)
    up22 = Dropout(0.2)(up22)
    up23 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up22)
    up31 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up23))
    up32 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up31)
    up32 = Dropout(0.2)(up32)
    up33 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up32)
    up41 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up33))
    up42 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up41)
    up42 = Dropout(0.2)(up42)
    up43 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up42)
    
    out = Conv2D(2, 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(up43)
    
    model = Model(inputs=[input_1,input_2],outputs=[out])
    return model

