import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from efficientnet.tfkeras import EfficientNetB0
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class Trainer:
    def __init__(self, epochs=100, batch_size=32, steps_per_epoch=100, validation_steps=50, learning_rate=0.001, patience=10, model_path='model.h5', train_dir='data/train', val_dir='data/val'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.learning_rate = learning_rate
        self.patience = patience
        self.model_path = model_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.build_model()

    def build_model(self):
        self.model = EfficientNetB0(include_top=True, weights=None, classes=1000)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
    
    def train(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_data = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical')

        val_data = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        
        self.model.fit(train_data,
                       steps_per_epoch=self.steps_per_epoch,
                       epochs=self.epochs,
                       validation_data=val_data,
                       validation_steps=self.validation_steps,
                       callbacks=[early_stopping, checkpoint])
    
    def main(self):
        self.train()

if __name__ == '__main__':
    train_dir = '/home/elias/clips/imagenet-mini/train'
    val_dir = '/home/elias/clips/imagenet-mini/val'
    
    trainer = Trainer(train_dir=train_dir, val_dir=val_dir)
    trainer.main()
