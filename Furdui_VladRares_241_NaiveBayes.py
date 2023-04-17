import numpy as np
from PIL import Image
from sklearn.naive_bayes import MultinomialNB
#importarea bibliotecilor


def valori_inspre_intervale(poze, intervale):
    poze_noi = np.zeros(poze.shape)
    for i in range(poze.shape[0]):
        poze_noi[i] = np.digitize(poze[i], intervale)
    return poze_noi - 1


# aceasta este o functie ce primeste un set de imagini si numarul de intervale si incadreaza aceste date
# intr-un numar de intervale dat
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
        i = i + 1
    return set


# aceasta functie are ca rol citirea de date, fie de validare, test sau train.
imagini_ct_train = citire_de_date(1, 17001)

imagini_ct_valid = citire_de_date(15001, 17001)

imagini_ct_test = citire_de_date(17001, 22150)
# apelarea functiei pentru cele trei seturi de date. Am citit si datele de valid in train pentru o antrenare cat mai eficienta

indici_train = []
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as f:
    f.readline()
    indici_train = [int(line.strip().split(",")[1]) for line in f][:15000]

indici_valid = []
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as f:
    f.readline()
    for line in f.readlines():
        indici_valid.append(int(line.strip().split(',')[1]))
        indici_train.append(int(line.strip().split(',')[1]))

# am citit labelurile pentru setul de train si valid, adaugand in indici_train si labelurile de la valid
acc = []
imagini_ct_valid = np.array((imagini_ct_valid))
imagini_ct_train = np.array((imagini_ct_train))
imagini_ct_test = np.array((imagini_ct_test))
intervale = np.linspace(0, 230, 4)

interval_train = valori_inspre_intervale(imagini_ct_train, intervale)
interval_train = interval_train.reshape(interval_train.shape[0], -1)
interval_valid = valori_inspre_intervale(imagini_ct_valid, intervale)
interval_valid = interval_valid.reshape(interval_valid.shape[0], -1)
interval_test = valori_inspre_intervale(imagini_ct_test, intervale)
interval_test = interval_test.reshape(interval_test.shape[0], -1)
#transformarea datelor in intervale si redimensionarea acestora
naive_bayes_model = MultinomialNB(alpha=3.2)
# apelarea functiei pentru Naive Bayes in care am introdus si un hiperparametru de 3.2
naive_bayes_model.fit(interval_train, indici_train)
# antrenarea modelului
print(naive_bayes_model.score(interval_valid, indici_valid))
indici_predictie = naive_bayes_model.predict(interval_test)

# apelarea functiei de predict pentru a obtine numarul de predictii
lista_afisare = [["id", "class"]]
for i in range(17001, 22150):
    lista_afisare.append([int(i), int(indici_predictie[i - 17001])])
# introducerea in lista_afisare datele pentru csv si ulterior salvarea fisierului
np.savetxt("Competitie.csv", lista_afisare, fmt="%s", delimiter=',')