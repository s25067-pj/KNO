import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image

def main():
    #baza danych
    mnist = tf.keras.datasets.mnist
    #przypisywanie do poszczególnych grup danych z bazy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #dzielenie przez 255 zeby uzyskac kolory szarosci, bez tego tez by sie obeszlo
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #tworzenie modelu
    model = tf.keras.models.Sequential([
        # 28x28 to dlugosc dwoch wektorow (rozmiar naszych obrazkow)
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # 128 -ilosc neuronow
        tf.keras.layers.Dense(128, activation='relu'),
        #zrobienie sztucznego szumu, zeby program sie nie przeuczyl
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    #jezeli jest zapisany model uczenia sie, uzyj go
    if os.path.exists('trained_model.h5'):
        model = tf.keras.models.load_model('trained_model.h5')
        print("Model został załadowany z pliku.")
    else:
        #jezeli nie, ucz sie
        model.fit(x_train, y_train, epochs=5)
        model.save('trained_model.h5')
        print("Model został zapisany do pliku.")

    e = model.evaluate(x_test, y_test)
    print(e)

    #uzyc negatywu, bedzie lepiej wychodzic
    img_path = "C:/Users/debis/PycharmProjects/KNO1/osiem.png"

    img = tf.keras.utils.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img_array = image.img_to_array(img)

    img_array = 255 - img_array  #negacja
    #img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    print(f"Predykcja: liczba to {predicted_class}")


if __name__ =='__main__':
    main()