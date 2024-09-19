from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPool2D, Conv2DTranspose, BatchNormalization, ReLU
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.models import Model
from keras.activations import sigmoid

def double_conv_block(signal, kernel, filters):
    output = Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=HeNormal())(signal)
    output = BatchNormalization()(output)
    output = ReLU()(output)
    output = Conv2D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer=HeNormal())(output)
    output = BatchNormalization()(output)
    output = ReLU()(output)
    return output

def downsample_block(signal, kernel, filters, downsample):
    f = double_conv_block(signal, kernel, filters)
    p = MaxPool2D(downsample)(f)
    return f, p

def upsample_block(signal, features, kernel, filters, upsample, padding='same'):
    output = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=upsample, padding=padding, kernel_initializer=HeNormal())(signal)
    output = BatchNormalization()(output)
    output = ReLU()(output)
    output = Concatenate()([output, features])
    output = double_conv_block(output, kernel, filters)
    return output

def create_model(shape=(252, 256, 10)):
    input=Input(shape=shape)
    f1, p1 = downsample_block(input, 5, 16, (1, 2))
    f2, p2 = downsample_block(p1, 3, 24, (1, 2))
    f3, p3 = downsample_block(p2, 3, 32, (1, 2))
    f4, p4 = downsample_block(p3, 3, 64, (1, 2))
    f5, p5 = downsample_block(p4, 3, 128, (1, 2))
    f6, p6 = downsample_block(p5, 3, 256, (1, 2))
    bottleneck = double_conv_block(p6, 3, 512)
    u8 = upsample_block(bottleneck, f6, 3, 256, (1, 2))
    u9 = upsample_block(u8, f5, 3, 128, (1, 2))
    u10 = upsample_block(u9, f4, 3, 64, (1, 2))
    u11 = upsample_block(u10, f3, 5, 32, (1, 2))
    u12 = upsample_block(u11, f2, 5, 24, (1, 2))
    u13 = upsample_block(u12, f1, 5, 16, (1, 2))

    u13 = Conv2DTranspose(1, (5, 1), strides=(1, 1), padding='valid')(u13)
    u13 = BatchNormalization()(u13)
    u13 = ReLU()(u13)

    output = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', kernel_initializer=GlorotUniform())(u13)
    output = BatchNormalization()(output)
    output = sigmoid(output)

    return Model(inputs = input, outputs = output)