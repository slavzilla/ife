from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Dense, BatchNormalization, Activation, Layer, Concatenate, Identity, Reshape, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Add
from tensorflow.keras.activations import swish, relu
import os
import json


import tensorflow as tf
class SqueezeExciteBlock(Layer):
    def __init__(self, num_filters, reduction, activation, initializer, sigmoid_initializer, **kwargs):

        assert isinstance(num_filters, int), "num_filters must be an integer."
        assert isinstance(reduction, int), "reduction must be an integer."
        assert activation in ['swish', 'relu'], "available activation functions are swish and relu."
        assert initializer in ['he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform', 'random_normal', 'random_uniform'], str(initializer)+" initializer not supported."
        assert sigmoid_initializer in ['he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform', 'random_normal', 'random_uniform'], str(sigmoid_initializer)+" initializer not supported."
        
        super(SqueezeExciteBlock, self).__init__(**kwargs)

        if activation == 'swish':
          self.activation = swish
        else:
          self.activation = relu

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(num_filters // reduction, activation=None, use_bias=False, kernel_initializer=initializer)
        self.activation = Activation(self.activation)
        self.dense2 = Dense(num_filters, activation='sigmoid', use_bias=False, kernel_initializer='glorot_normal') # mozda sa sigmoidom bolje da ide Glorot inicijalizacija, a sa relu i swish he
        self.reshape = Reshape((1, 1, num_filters))

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        scale = self.reshape(x)
        return inputs * scale

class DenseBlock(Layer):
    def __init__(self, num_layers, num_filters, num_groups, filter_size, bottleneck, bottleneck_reduction, squeeze, squeeze_reduction, activation, initializer, **kwargs):

        assert isinstance(num_layers, int), "num_layers must be an integer."
        assert isinstance(num_filters, int), "num_filters must be an integer."
        assert isinstance(num_groups, int), "num_groups must be an integer."
        assert isinstance(filter_size, int), "filter_size must be an integer."
        assert isinstance(bottleneck, bool), "bottleneck must be a bool value."
        assert isinstance(bottleneck_reduction, int), "bottleneck_reduction must be an integer"
        assert isinstance(squeeze, bool), "squeeze must be a bool value."
        assert isinstance(squeeze_reduction, int), "squeeze_reduction must be am integer."
        assert activation in ['swish', 'relu'], "available activation functions are swish and relu."
        assert initializer in ['he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform', 'random_normal', 'random_uniform'], str(initializer)+" initializer not supported."
            
        super(DenseBlock, self).__init__(**kwargs)

        if activation == 'swish':
            self.activation = swish
        else:
            self.activation = relu

        self.bottleneck_batchnorm_layers = []
        self.bottleneck_activation_layers = []
        self.bottleneck_conv_layers = []

        self.batchnorm_layers = []
        self.activation_layers = []
        self.conv_layers = []

        for _ in range(num_layers):
            if bottleneck is True: # Da li pred svaku konvoluciju da smanjuje broj ulaznih kanala. U originalnom radu su smanjivali na 4*k, k je growth rate (num_filters otp).
                self.bottleneck_batchnorm_layers.append(BatchNormalization())
                self.bottleneck_activation_layers.append(Activation(self.activation))
                self.bottleneck_conv_layers.append(Conv2D(num_filters//bottleneck_reduction, kernel_size=1, padding='same', groups=num_groups, kernel_initializer=initializer))
            else:
                self.bottleneck_batchnorm_layers.append(Identity())
                self.bottleneck_activation_layers.append(Identity())
                self.bottleneck_conv_layers.append(Identity())

            self.batchnorm_layers.append(BatchNormalization())
            self.activation_layers.append(Activation(self.activation))
            self.conv_layers.append(Conv2D(num_filters, filter_size, padding='same', groups=num_groups, kernel_initializer=initializer))

        self.se = Identity()
        if squeeze is True:
            self.se = SqueezeExciteBlock(num_filters*(num_layers+1), squeeze_reduction, activation, initializer, initializer)
    
    def call(self, inputs):
        x = inputs
        for btnck_bn, btnck_ac, btnck_conv, bn, activation, conv in zip(self.bottleneck_batchnorm_layers, self.bottleneck_activation_layers, self.bottleneck_conv_layers, self.batchnorm_layers, self.activation_layers, self.conv_layers):
            x1 = btnck_bn(x)
            x1 = btnck_ac(x1)
            x1 = btnck_conv(x1)
            x1 = conv(x1)
            x1 = bn(x1)
            x1 = activation(x1)
            x = Concatenate()([x, x1])
    
        x = self.se(x)
        return x
  
class TransitionBlock(Layer):
    def __init__(self, num_filters, num_groups, conv_layers, pooling, pooling_type, activation, initializer, **kwargs):

        assert isinstance(num_filters, int), "num_filters must be an integer."
        assert isinstance(num_groups, int), "num_groups must be an integer."
        assert isinstance(pooling, bool), "pooling must be a bool value."
        assert conv_layers in [0, 1, 2], "conv_layers must be 0, 1 or 2."
        assert pooling is True or conv_layers > 0, "pooling="+str(pooling)+" and conv_layers="+str(conv_layers)+" not supported."
        assert pooling_type in ['avg', 'max', None], "available pooling_types are max, avg and None."
        assert activation in ['swish', 'relu'], "available activation functions are swish and relu."
        assert initializer in ['he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform', 'random_normal', 'random_uniform'], str(initializer)+" initializer not supported."

        super(TransitionBlock, self).__init__(**kwargs)

        if activation == 'swish':
            self.activation = swish
        else:
            self.activation = relu

        self.batch_norm_layers = []
        self.activation_layers = []
        self.conv_layers = []

        self.pooling_layer = Identity()

        if pooling is True: #ako hocemo pooling
            if pooling_type == 'max':
                self.pooling_layer = MaxPooling2D(pool_size=2)
            else:
                self.pooling_layer = AveragePooling2D(pool_size=2)

        if conv_layers == 0:
            self.conv_layers.append(Identity())
            self.conv_layers.append(Identity())
            self.batch_norm_layers.append(Identity())
            self.batch_norm_layers.append(Identity())
            self.activation_layers.append(Identity())
            self.activation_layers.append(Identity())
        elif conv_layers == 1 and pooling is True:
            self.conv_layers.append(Conv2D(num_filters, groups=num_groups, kernel_size=1, strides=1))
            self.conv_layers.append(Identity())
            self.batch_norm_layers.append(BatchNormalization())
            self.batch_norm_layers.append(Identity())
            self.activation_layers.append(Activation(self.activation))
            self.activation_layers.append(Identity())
        elif conv_layers == 1 and pooling is False:
            self.conv_layers.append(Conv2D(num_filters, groups=num_groups, kernel_size=1, strides=2))
            self.conv_layers.append(Identity())
            self.batch_norm_layers.append(BatchNormalization())
            self.batch_norm_layers.append(Identity())
            self.activation_layers.append(Activation(self.activation))
            self.activation_layers.append(Identity())
        else:
            self.conv_layers.append(Conv2D(num_filters, groups=num_groups, kernel_size=1, strides=1, kernel_initializer=initializer))
            self.conv_layers.append(Conv2D(num_filters, groups=num_groups, kernel_size=1, strides=2, kernel_initializer=initializer))
            self.batch_norm_layers.append(BatchNormalization())
            self.batch_norm_layers.append(BatchNormalization())
            self.activation_layers.append(Activation(self.activation))
            self.activation_layers.append(Activation(self.activation))
    
    def call(self, inputs):
        x = inputs
        for bn, activation, conv in zip(self.batch_norm_layers, self.activation_layers, self.conv_layers):
            x = bn(x)
            x = activation(x)
            x = conv(x)

        x = self.pooling_layer(x)

        return x



class DenseNet(Model):
    def __init__(self, initial_filters=32, initial_filter_size=3, initial_stride=1, num_dense_blocks=4, num_layers_per_block=[6, 12, 24, 16], num_filters=32, num_groups=1, filter_size=3, bottleneck=False, bottleneck_reduction=4, squeeze=False, squeeze_reduction=16, pooling=False, pooling_type=None, transition_conv_layers=1, activation='relu', initializer='he_normal', **kwargs):

        assert isinstance(initial_filters, int), "initial_filters must be an integer."
        assert isinstance(initial_filter_size, int), "initial_filter_size must be an integer."
        assert isinstance(initial_stride, int), "initial_stride must be an integer."
        assert isinstance(num_dense_blocks, int), "num_dense_blocks must be an integer."
        assert isinstance(num_layers_per_block, int) or num_dense_blocks == len(num_layers_per_block), "num_layers_per_block must be an integer or have num_dense_blocks elements."
        assert not(pooling is True and num_dense_blocks==1), "pooling not supported with one dense block."
        super(DenseNet, self).__init__(**kwargs)

        if isinstance(num_layers_per_block, int):
            num_layers_per_block = [num_layers_per_block]*num_dense_blocks

        if activation == 'swish':
            self.activation = swish
        else:
            self.activation = relu

        self.initial_conv = Conv2D(initial_filters, initial_filter_size, initial_stride, groups=num_groups, padding='same', kernel_initializer=initializer)
        self.initial_batchnorm = BatchNormalization()
        self.initial_activation = Activation(self.activation)
        self.initial_pool = Identity()

        if pooling is True:
            if pooling_type == 'max':
                self.initial_pool = MaxPooling2D(pool_size=2)
            else:
                self.initial_pool = AveragePooling2D(pool_size=2)

        self.dense_blocks = []
        self.transition_blocks = []
        for i in range(num_dense_blocks):
            self.dense_blocks.append(DenseBlock(num_layers_per_block[i], num_filters, num_groups, filter_size, bottleneck, bottleneck_reduction, squeeze, squeeze_reduction, activation, initializer))
            if i != num_dense_blocks - 1:
                self.transition_blocks.append(TransitionBlock(num_filters, num_groups, transition_conv_layers, pooling, pooling_type, activation, initializer))

        #self.final_bn = BatchNormalization()
        #self.final_activation = Activation(activation)
    def call(self, inputs):
        x = inputs
        x = self.initial_conv(inputs)
        x = self.initial_batchnorm(x)
        x = self.initial_activation(x)
        x = self.initial_pool(x)
        for dense_block, transition_block in zip(self.dense_blocks, self.transition_blocks + [None]):
            x = dense_block(x)
            if transition_block:
                x = transition_block(x)

        #x = self.final_bn(x)
        #x = self.final_activation(x)
        return x

class DenseNetAE(DenseNet):
    def __init__(self, initial_filters=32, initial_filter_size=11, initial_stride=1, num_dense_blocks=4, num_layers_per_block=[6, 12, 24, 16], num_filters=32, num_groups=1, filter_size=3, bottleneck=False, bottleneck_reduction=4, squeeze=False, squeeze_reduction=16, pooling=False, pooling_type=None, transition_conv_layers=1, activation='relu', final_activation='sigmoid', initializer='he_normal', final_initializer='glorot_uniform', **kwargs):
        super(DenseNetAE, self).__init__(initial_filters, initial_filter_size, initial_stride, num_dense_blocks, num_layers_per_block, num_filters, num_groups, filter_size, bottleneck, bottleneck_reduction, squeeze, squeeze_reduction, pooling, pooling_type, transition_conv_layers, activation, initializer, **kwargs)
        self.batchnorm = BatchNormalization()
        self.conv = Conv2D(1, 1, padding='same', kernel_initializer=final_initializer)
        self.activation = Activation(activation)
        
        self.final_batchnorm = BatchNormalization()
        self.upsample = Conv2DTranspose(1, (5, 1), strides=1, padding='valid')
        self.final_activation = Activation(final_activation)
    def call(self, inputs):
        x = super().call(inputs)
        
        x = self.batchnorm(x)
        x = self.conv(x)
        x = self.activation(x)

        x = self.final_batchnorm(x)
        x = self.upsample(x)
        x = self.final_activation(x)

        return x
    

def create_model():
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "dense_config.json")
    file = open(config_path)
    metadata = json.load(file)
    return DenseNetAE(**metadata["architecture_params"])