import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime


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
    y = to_categorical(y-1)
    x = df.drop(columns=['Class']).values

    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model1.fit(x_train, y_train, epochs=50, batch_size=8, callbacks=[tensorboard_callback],
               validation_data=(x_test, y_test))

    model2 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.01)

    model2.compile(optimizer=optimizer1, loss='categorical_crossentropy', metrics=['accuracy'])

    model2.fit(x_train, y_train, epochs=50, batch_size=8, callbacks=[tensorboard_callback1],
               validation_data=(x_test, y_test))

    loss1, accuracy1 = model1.evaluate(x_test, y_test)
    loss2, accuracy2 = model2.evaluate(x_test, y_test)

    print(f"Model 1 - Loss: {loss1}, Accuracy: {accuracy1}")
    print(f"Model 2 - Loss: {loss2}, Accuracy: {accuracy2}")

    e1 = model1.evaluate(x_test, y_test)
    print(e1)

    e2 = model2.evaluate(x_test, y_test)
    print(e2)


if __name__ == "__main__":
    main()
