from keras.models import *
from keras.layers import *

def GLASPP():
    #GLASPP function def

def self_attention(b5):
    w,x,y,z = b5.shape
    z1=int(z/8)
    query = Conv2D(z1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b5)            #can change the factor to divide 
    key = Conv2D(z1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b5)
    value = Conv2D(z, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(b5)

    query = Reshape([x*y, z1])(query)
    key = Reshape([z1, x*y])(key)
    value = Reshape([z, x*y])(value)

    prod = Dot(axes = 2)([query,Permute(dims = (2,1))(key)])
    w = z1**-2
    prod = Lambda(lambda x: x * w)(prod)
    prod = Softmax()(prod)
    
    attention = Dot(axes = 2)([value,Permute(dims = (2,1))(prod)])
    attention = Reshape([x,y,z])(attention)
    #add attention gate instead of the add function
    attention = Add()([attention,b5])
    return(attention)