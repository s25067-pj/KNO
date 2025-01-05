import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime
import keras_tuner as kt


class WineModel(tf.keras.Model):
    def __init__(self, hp):
        super(WineModel, self).__init__()

        self.num_layers = hp.Int('num_layers', min_value=1, max_value=10, step=1)
        self.units = hp.Int('units', min_value=32, max_value=1024, step=32)
        self.learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])

        self.hidden_layers = [
            tf.keras.layers.Dense(self.units, activation='relu') for _ in range(self.num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(3, activation='softmax')
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


def model_builder(hp):
    return WineModel(hp)


def main():
    logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    input_file = 'wine.data'
    column_names = [
        'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
        'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
        'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
    ]
    df = pd.read_csv(input_file, header=None, names=column_names)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df['Class'].values
    y = to_categorical(y - 1)
    x = df.drop(columns=['Class']).values
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(x)

    tuner = kt.Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=10,
        factor=3,
        directory='tuner_logs',
        project_name='wine_classification_with_multiple_layers'
    )

    tuner.search(X_standardized, y, epochs=100, batch_size=64, validation_data=(X_standardized,y), callbacks=[tensorboard_callback])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nNajlepsze hiperparametry:")
    print(f"Liczba warstw: {best_hps.get('num_layers')}")
    print(f"Liczba neuronów w warstwie: {best_hps.get('units')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    best_model = tuner.hypermodel.build(best_hps)

    best_model.fit(X_standardized, y, epochs=100, batch_size=64, callbacks=[tensorboard_callback])

    loss, accuracy = best_model.evaluate(X_standardized, y)
    print(f"\nTest Loss: {loss}, Test Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
