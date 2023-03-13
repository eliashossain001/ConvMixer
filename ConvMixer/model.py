import tensorflow as tf
from tensorflow.keras import layers
import os 

#define the convmixer class

class ConvMixer(tf.keras.Model):
    def __init__(self, num_classes):
        super(ConvMixer, self).__init__()
        self.conv1 = layers.Conv2D(filters=256, kernel_size=(9, 9), strides=(4, 4), padding="same")
        self.depthwise_layers = []
        self.pointwise_layers = []
        
        for _ in range(4):
            self.depthwise_layers.append(layers.DepthwiseConv2D(kernel_size=9, strides=(1, 1), padding="same"))
            self.pointwise_layers.append(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same"))
            # self.pointwise_layers.append(layers.BatchNormalization())
            self.pointwise_layers.append(layers.BatchNormalization(epsilon=1e-5))
            self.pointwise_layers.append(layers.Activation("relu"))
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(units=num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        for depthwise, pointwise in zip(self.depthwise_layers, self.pointwise_layers):
            x = depthwise(x)
            x = pointwise(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x
