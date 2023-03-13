import os
import tensorflow as tf
from setup import INPUT_SHAPE, NUM_CLASSES, TRAIN_DIR, VAL_DIR, BATCH_SIZE, EPOCHS
from model import ConvMixer
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ConvMixerTrainer:
    def __init__(self, input_shape, num_classes, train_dir, val_dir, batch_size, epochs):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.train_generator = None
        self.val_generator = None

    def create_model(self):
        self.model = ConvMixer(self.num_classes)

    def compile_model(self):
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def create_checkpoint(self):
        checkpoint_dir = './checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')

        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

    def train_model(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        self.train_generator = datagen.flow_from_directory(
            self.train_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        self.val_generator = datagen.flow_from_directory(
            self.val_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=[self.checkpoint_callback]
        )


def main():
    trainer = ConvMixerTrainer(INPUT_SHAPE, NUM_CLASSES, TRAIN_DIR, VAL_DIR, BATCH_SIZE, EPOCHS)
    trainer.create_model()
    trainer.compile_model()
    trainer.create_checkpoint()
    trainer.train_model()


if __name__ == '__main__':
    main()
