import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


def main():
    directory = "zwierzeta"

    zwierzeta = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=64,
        image_size=(128, 128),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True
    )

    x_train = []
    for images, _ in zwierzeta:
        x_train.append(images.numpy())

    x_train = np.concatenate(x_train, axis=0)

    x_train = x_train.astype('float32') / 255.0

    print(x_train.shape)

    class Autoencoder(Model):
        def __init__(self, latent_dim, shape):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = tf.keras.Sequential([
                layers.Conv2D(8, 5, data_format="channels_last", activation='relu'),
                layers.Conv2D(8, 3, data_format="channels_last", activation='relu'),
                layers.Flatten(),
                #layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='relu'),
                layers.Dense(latent_dim),
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(8, activation='relu'),
                #layers.Dropout(0.2),
                layers.Dense(128 * 128 * 3, activation='sigmoid'),
                layers.Reshape([128, 128, 3]),
                layers.Reshape(shape)
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    x_train = tf.reshape(x_train, (len(x_train), 128, 128, 3))
    shape = x_train.shape[1:]
    latent_dim = 2
    autoencoder = Autoencoder(latent_dim, shape)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(x_train, x_train,
                    epochs=400,
                    shuffle=True,
                    batch_size=8)

    encoded_imgs = tf.constant([[[x / 5.0 - 1.0, y / 5.0 - 1.0] for x in range(5)] for y in range(5)])
    encoded_imgs = tf.reshape(encoded_imgs, (5 * 5, 2))

    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    fig, axs = plt.subplots(5, 5, figsize=(5, 5))

    # Plot in each subplot
    for y in range(5):
        for x in range(5):
            axs[y, x].imshow(decoded_imgs[y * 5 + x])
            axs[y, x].axis('off')

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
