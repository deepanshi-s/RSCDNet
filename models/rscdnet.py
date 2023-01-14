from keras.models import *
from keras.layers import *
from keras import backend as K
import tensorflow as tf


def gating(layer):
    weights = GlobalAveragePooling2D()(layer)
    return Multiply()([weights,layer])


def GLASPP(CONV5):
    #GLASPP function def
    b1 = Conv2D(256, 3, dilation_rate=(4, 4), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name ='b1' )(CONV5)
    b1 = Dropout(0.2)(b1)
    b1 = gating(b1)

    b6 = Conv2D(256, 3, dilation_rate=(6, 6), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name ='b6' )(CONV5)
    b6 = Dropout(0.2)(b6)
    b6 = gating(b6)

    #rate 12
    b2 = Conv2D(256, 3, dilation_rate=(8, 8), activation='relu', padding = 'same', kernel_initializer = 'he_normal' , name = 'b2')(CONV5)
    b2 = Dropout(0.2)(b2)
    b2 = gating(b2)

    #rate 18
    b3 = Conv2D(256, 3, dilation_rate=(12, 12), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name = 'b3' )(CONV5)
    b3 = Dropout(0.2)(b3)
    b3 = gating(b3)

    #rate 24
    b4 = Conv2D(256, 3, dilation_rate=(16, 16), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name = 'b4' )(CONV5)
    b4 = Dropout(0.2)(b4)
    b4 = gating(b4)
    
    b7 = Conv2D(256, 3, dilation_rate=(18, 18), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name ='b7' )(CONV5)
    b7 = Dropout(0.2)(b7)
    b7 = gating(b7)

    b5 = Concatenate()([b1, b2, b3, b4, b6, b7])
    b5_tanh = Activation(tf.keras.activations.tanh)(b5)
    b5_sigmoid = Activation(tf.keras.activations.sigmoid)(b5)
    b5 = Multiply()([b5_tanh, b5_sigmoid])   
    b5 = Conv2D(256, 1, padding = 'same', kernel_initializer = 'he_normal')(b5)
    
    return b5


def attentionlayer(skip_layer,down_layer):
    inter_channel = skip_layer.get_shape().as_list()[3]
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(down_layer)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(skip_layer)
    added = Activation('relu')(add([theta_x, phi_g]))
    psi = Conv2D(1, [1, 1], strides=[1, 1],activation="sigmoid")(added)
    multiplied = multiply([skip_layer, psi])
    return multiplied


def self_attention(b5):
    w,x,y,z = b5.shape
    z1=int(z/8)
    query = Conv2D(z1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b5)            #can change the factor to divide 
    key = Conv2D(z1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b5)
    value = Conv2D(z, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b5)
    value = gatting2(value, b5)

    query = Reshape([x*y, z1])(query)
    key = Reshape([z1, x*y])(key)
    value = Reshape([z, x*y])(value)

    prod = Dot(axes = 2)([query,Permute(dims = (2,1))(key)])
    w = z1**-2
    prod = Lambda(lambda x: x * w)(prod)
    prod = Softmax()(prod)
    
    attention = Dot(axes = 2)([value, Permute(dims = (2,1))(prod)])
    attention = Reshape([x,y,z])(attention)
    attention = attentionlayer(b5,attention)
    attention = Add()([attention,b5])
    return(attention)



def siam_model(input_shape):
    
    input_size=(input_shape)

    conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv1_2 = Dropout(0.2)
    conv1_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    
    pool1_1 = MaxPooling2D(pool_size=(2, 2))
    conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv2_2 = Dropout(0.2)
    conv2_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    
    pool2_1 = MaxPooling2D(pool_size=(2, 2))
    conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv3_2 = Dropout(0.2)
    conv3_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    
    pool3_1 = MaxPooling2D(pool_size=(2, 2))
    conv4_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv4_2 = Dropout(0.2)
    conv4_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    
    input1 = Input(input_size)
    aconv1_1 = conv1_1(input1)
    aconv1_2 = conv1_2(aconv1_1)
    aconv1_3 = conv1_3(aconv1_2)
    
    apool1_1 = pool1_1(aconv1_3)
    aconv2_1 = conv2_1(apool1_1)
    aconv2_2 = conv2_2(aconv2_1)
    aconv2_3 = conv2_3(aconv2_2)
    
    apool2_1 = pool2_1(aconv2_3)
    aconv3_1 = conv3_1(apool2_1)
    aconv3_2 = conv3_2(aconv3_1)
    aconv3_3 = conv3_3(aconv3_2)
    
    apool3_1 = pool3_1(aconv3_3)
    aconv4_1 = conv4_1(apool3_1)
    aconv4_2 = conv4_2(aconv4_1)
    aconv4_3 = conv4_3(aconv4_2)
    
    input2 = Input(input_size)
    bconv1_1 = conv1_1(input2)
    bconv1_2 = conv1_2(bconv1_1)
    bconv1_3 = conv1_3(bconv1_2)
    
    bpool1_1 = pool1_1(bconv1_3)
    bconv2_1 = conv2_1(bpool1_1)
    bconv2_2 = conv2_2(bconv2_1)
    bconv2_3 = conv2_3(bconv2_2)
    
    bpool2_1 = pool2_1(bconv2_3)
    bconv3_1 = conv3_1(bpool2_1)
    bconv3_2 = conv3_2(bconv3_1)
    bconv3_3 = conv3_3(bconv3_2)
    
    bpool3_1 = pool3_1(bconv3_3)
    bconv4_1 = conv4_1(bpool3_1)
    bconv4_2 = conv4_2(bconv4_1)
    bconv4_3 = conv4_3(bconv4_2)
    
    
    
    subtracted = Subtract(name = 'subtract')([aconv4_3, bconv4_3])


    CONV5 = self_attention(subtracted)
    
    glasppOutput = GLASPP(CONV5)
    
    added = Add()([glasppOutput,subtracted])

    #self
    up21 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(added))
    up22 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up21)
    up22 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up22)

    up22 = Dropout(0.2)(up22)
    up23 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up22)
    up31 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up23))
    up32 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up31)
    up32 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up32)

    up32 = Dropout(0.2)(up32)
    up33 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up32)
    up41 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up33))
    up42 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up41)
    up42 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up42)

    up42 = Dropout(0.2)(up42)
    up43 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up42)
    
    out = Conv2D(2, 1, padding = 'same', activation = 'softmax',kernel_initializer = 'he_normal')(up43)
    
    model = Model(inputs=[input1,input2],outputs=[out])
    print(model.summary())
    return model

final_model = siam_model((512,512,3))