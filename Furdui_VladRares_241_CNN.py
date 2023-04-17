import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#importare de biblioteci


def citire_de_date(st, sf):
    i = st
    set = []
    while i < sf:
        if i < 10:
            cale = "/kaggle/input/unibuc-brain-ad/data/data/00000" + str(i) + ".png"
        elif i < 100:
            cale = "/kaggle/input/unibuc-brain-ad/data/data/0000" + str(i) + ".png"
        elif i < 1000:
            cale = "/kaggle/input/unibuc-brain-ad/data/data/000" + str(i) + ".png"
        elif i < 10000:
            cale = "/kaggle/input/unibuc-brain-ad/data/data/00" + str(i) + ".png"
        else:
            cale = "/kaggle/input/unibuc-brain-ad/data/data/0" + str(i) + ".png"
        poza = Image.open(cale).convert('L')
        linie = np.array(poza)
        linie_np = np.array(linie).flatten()
        set.append(linie)
        i += 1

    return set
#citirea de elemente si returnarea unei liste

imagini_ct_train = citire_de_date(1, 15001)
print("train")
#citirea imaginilor pentru antrenament
imagini_ct_valid = citire_de_date(15001, 17001)
print("valid")
#citirea imaginilor pentru valiare
imagini_ct_test = citire_de_date(17001, 22150)
print("test")
#citirea imaginilor pentru test
indici_train = []

with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as f:
    f.readline()
    indici_train = [int(line.strip().split(",")[1]) for line in f][:15000]

indici_valid = []

with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as f:
    f.readline()
    for line in f.readlines():
        indici_valid.append(int(line.strip().split(',')[1]))          #citirea label-urilor
#crearea modelului prin Sequential
model = Sequential([


    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),  #strat de tip convolutionar
    MaxPooling2D(),                #strat care reduce datele


    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(),


    Flatten(),  #layer pentru transformarea rezultatului
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),   #layere conectate total la rezultat
    Dropout(0.25),                  #25% din rezultat va fi 0 selectat random
    Dense(1, activation='sigmoid')])    #rezultatul apartenentei la o clasa
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(np.array(imagini_ct_train).reshape(-1, 224, 224, 1), np.array(indici_train), epochs=32,
                    batch_size=32,
                    validation_data=(np.array(imagini_ct_valid).reshape(-1, 224, 224, 1), np.array(indici_valid)))   #antrenare cu 32 de epoci si imparire in 32
indici_predictie = model.predict(np.array(imagini_ct_test).reshape(-1, 224, 224, 1))  #prezicerea rezultatelor
print(indici_predictie)
for i in range(len(indici_predictie)):
    if indici_predictie[i] < 0.5:
        indici_predictie[i] = 0
    else:
        indici_predictie[i] = 1
    print(indici_predictie[i])
#transformarea rezultatelor in 0 si 1
lista_afisare = [["id", "class"]]
for i in range(17001, 22150):
    lista_afisare.append([int(i), int(indici_predictie[i - 17001])])
np.savetxt("Competitie.csv", np.array(lista_afisare), fmt="%s", delimiter=',')
#salvarea lor ca csv