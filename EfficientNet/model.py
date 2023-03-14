from setup import NUM_CLASSES
import tensorflow as tf 
from efficientnet.tfkeras import EfficientNetB0

DEFAULT_NUM_CLASSES = 10

class MyEfficientNet:
    
    def __init__(self, input_shape, num_classes=DEFAULT_NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_model(self):
        base_model = EfficientNetB0(input_shape=self.input_shape,
                                    include_top=False,
                                    weights='imagenet')

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

        for layer in base_model.layers:
            layer.trainable = False

        return model
