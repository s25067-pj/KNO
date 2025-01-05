import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime


def model_for_params(units, learning_rate=0.01):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(13,)))

    for neurons in units:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))

    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


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
    y = to_categorical(y-1)  # One-hot encoding
    x = df.drop(columns=['Class']).values

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.7, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    baseline_config = [32, 16]
    baseline_learning_rate = 0.01

    model = model_for_params(baseline_config, baseline_learning_rate)
    model.fit(x_train, y_train, epochs=50, batch_size=8, callbacks=[tensorboard_callback],
              validation_data=(x_val, y_val))

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Model - Loss: {loss}, Accuracy: {accuracy}")

    units = [[32, 16], [64, 32, 16]]
    learning_rates = [0.01, 0.001]
    batch_sizes = [16, 32]

    best_accuracy = 0
    best_params = None
    best_model = None
    results = []

    for rates in learning_rates:
        for unit in units:
            for batch in batch_sizes:
                actual_model = model_for_params(unit, rates)
                actual_model.fit(x_train, y_train, epochs=50, batch_size=batch, callbacks=[tensorboard_callback],
                                    validation_data=(x_val, y_val))
                actual_loss, actual_accuracy = actual_model.evaluate(x_val, y_val)
                print(f"Actual model (learning rates: {rates}, units: {unit}, batch size: {batch}) - loss: {actual_loss}, accuracy: {actual_accuracy}")

                results.append({
                    'learning_rate': rates,
                    'units': unit,
                    'batch_size': batch,
                    'val_loss': actual_loss,
                    'val_accuracy': actual_accuracy
                })

                if actual_accuracy > best_accuracy:
                    best_accuracy = actual_accuracy
                    best_params = (rates, unit, batch)
                    best_model = actual_model

    print("\nAll results:")
    actual_results = pd.DataFrame(results)
    print(actual_results)

    print("\nBest parameters found:")
    print(f"Learning Rate: {best_params[0]}, Architecture: {best_params[1]}, Batch Size: {best_params[2]}")
    print(f"Validation Accuracy: {best_accuracy}")

    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    return actual_results, best_params, test_loss, test_accuracy


if __name__ == "__main__":
    main()
