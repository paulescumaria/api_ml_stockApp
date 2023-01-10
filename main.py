import json
import pandas as pd

from numpy import array
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import datetime
import math
import sys
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# ne conectam la baza de date Firebase cu credentialele aferente
def firebase_initialize():
    cred = credentials.Certificate("appstock-fe44a-f92cf5e7f6b4.json")
    app = firebase_admin.initialize_app(cred)

# ne conectam la clientul de firestore pentru a avea acces la documente si colectii
def firebase_connection():
    db = firestore.client()
    return db

# extragem datele pentru prezicere si antrenament din firestore
def return_data(id_preparat):
    # conectam scriptul la baza de date firestore
    db = firebase_connection()
    result_data = []
    if db:
        #extragem toate datele din colectia orders-complete
        response = db.collection(u'orders-complete').stream()
        if response:
            for data in response:
                result_data.append({u'id': data.id, u'products': data.to_dict()})  # aducem datele sub forma unui json pentru a putea sa le manipulam mai usor
    restaurant_dat_x = []
    restaurant_dat_y = []
    restaurant_dat_predict_x = []
    restaurant_dat_predict_y = []

    if len(result_data) == 0:
        return [], [], [], []

    today = datetime.date.today()
    first = today.replace(day=1)
    lastMonth = first - datetime.timedelta(days=1)
    # construim dataset-ul pentru axa x unde folosim valoarea 1, pe viitor se poate pune valori precum nr. de persoane, in ce tip de anotimp suntem [1]
    # construim dataset-ul pentru axa y unde folosim coloana ['Cantitate']
    # axa y va fii axa care va fii prezisa
    # prezicem cantitatea pentru a putea stii cum sa ne facem stocul pentru luna viitoare
    # am compus axa x pentru a realiza o prezicere mai buna, pe viitor variabila independenta fiind compusa de exemplu
    # din coloanele nr_persoane, tip anotimp ar oferii o prezicere si mai exacta pentru restaurant 
    for row in result_data:
        # iteram prin cheile json-ului din products, de exemplu {Pizza: 3, Salata: 2}, iteram prin cheile Pizza, Salata
        json_to_string = json.dumps(row['products'])
        data_keys = json.loads(json_to_string)
        for product in data_keys:
            # daca cheia curenta este egala cu preparatul pentru care vrem sa prezicem ne apucam sa construim dataset-urile
            # gen daca cheia Pizza === cu id_preparat care este tot Pizza
            if product == id_preparat:
                # luam datele din ultima luna pentru antrenament
                # iar datele din ultima luna pentru ajustarea erorii de prezicere
                # vom prezice datele in fiecare data de 25 a fiecarei luni
                if row['id'] < lastMonth.strftime("%m-01-%Y"):
                    restaurant_dat_x.append([1])
                    restaurant_dat_y.append([row['products'][product]])
                else:
                    restaurant_dat_predict_x.append([1])
                    restaurant_dat_predict_y.append([row['products'][product]])

    if len(restaurant_dat_x) == 0:
        restaurant_dat_x.append([1])

    if len(restaurant_dat_y) == 0:
        restaurant_dat_y.append([1])
    # dupa ce am construit dataset-urile din baza de date transformam lista in array pentru a putea trimite
    # dataset-urile ca si parametrii
    restaurant_dat_x = array(restaurant_dat_x)
    restaurant_dat_y = array(restaurant_dat_y)
    restaurant_dat_predict_x = array(restaurant_dat_predict_x)
    restaurant_dat_predict_y = array(restaurant_dat_predict_y)
    return restaurant_dat_x, restaurant_dat_y, restaurant_dat_predict_x, restaurant_dat_predict_y


def init(restaurant_x_train, restaurant_y_train):
    # xgbregressor vine de la Extreme Gradient Boost Regressor si am folosit acest algoritm
    # pentru precizia pe care o are, cu o cantitate mica de date putem sa facem o precizie
    # de 1 si o eroare de zero, care este foarte buna
    # acest algoritm ofera o precizie foarte buna deoarece foloseste arbori de decizie la care pentru fiecare nou calcul pe care il
    # efectueaza algoritmul foloseste rezultatul anterior numai daca indeplineste un set de criterii, astfel eroarea se micsoreaza cu fiecare
    # raspuns anterior folosit, este o combinație perfectă de tehnici de optimizare software și hardware pentru a obține rezultate superioare
    # folosind mai puține resurse de calcul în cel mai scurt timp.
    regressor = xgb.XGBRegressor()
    # antrenam algoritmul cu dataset-urile de antrenare
    regressor.fit(restaurant_x_train, restaurant_y_train)
    return regressor


def xgboost_regression(id_preparat):

    # returnam datele din documentul orders-complete in dataset-urile aferente
    restaurant_dat_x, restaurant_dat_y, restaurant_dat_predict_x, restaurant_dat_predict_y = return_data(id_preparat)

    if len(restaurant_dat_predict_x) != 0 and len(restaurant_dat_predict_y) != 0:
        # folosim aceasta functie pentru a imparti dataset-ul in doua parti, cel care va fii folosit
        # pentru testare si cel pentru antrenare
        # folosim test_size=0.8 pentru a folosii 80% pentru antrenare si 20% pentru testare
        # foloseam aceasta functie atunci cand voiam sa afisam sub forma grafica cu plot
        # nu mai folosim aceasta functie ca avem deja datele impartite cum trebuie
        # restaurant_x_test, restaurant_x_train, restaurant_y_test, restaurant_y_train = train_test_split(restaurant_dat_x, restaurant_dat_y, test_size=0.8, random_state=0)

        # initializam si antrenam algortimul
        regressor = init(restaurant_dat_x, restaurant_dat_y)

        # prezicem cantitatea pentru viitor pe baza dataset-ului de test
        restaurant_y_pred = regressor.predict(restaurant_dat_predict_x)

        # printam eroarea, coeficientul de predictie si dataset-ul cu cantatitatea pe viitor consumata
        arithmetic_average = 0
        for quantity in restaurant_y_pred[:31]:
            arithmetic_average += quantity
        arithmetic_average /= len(restaurant_y_pred[:31])
        arithmetic_average += mean_squared_error(restaurant_dat_predict_y, restaurant_y_pred)
       
        db = firebase_connection()
        result_data = []
        # cautam sa vedem daca nu cumva am mai prezis pentru data curenta
        response = db.collection(u'stock-prediction').stream()
        if response:
            for data in response:
                result_data.append({u'id': data.id, u'products': data.to_dict()})

        if len(result_data) > 0:
            result_data = list(filter(
                lambda x: x['id'] == datetime.date.today().strftime("%m-%d-%Y") and id_preparat in json.loads(
                    json.dumps(x['products'])), result_data))

        # daca lungimea listei este zero, inseamna ca nu mai exista o prezicere pentru data si preparatul respectiv
        if len(result_data) == 0:
            # inseram in baza de date valoarea prezisa pentru preparatul curent si la data curenta
            response = db.collection(u'stock-prediction').document(datetime.date.today().strftime("%m-%d-%Y")).set({
                id_preparat: math.ceil(arithmetic_average)
            }, merge=True)

            # daca inserarea a avut loc cu succes printam 1
            if response:
                print(1)
            else:
                # daca a avut loc o eroare printam -1
                print(-1)
        else:
            # printam 2 daca avem deja o valoare prezisa pentru preparatul curent
            print(2)

    else:
        # printam 0 daca nu avem date suficiente pentru prezis
        print(0)
        # dupa ce am prezis cantitatea pentru cele 30-31 de zile facem o medie pentru toata luna, adunam cu eroarea pentru a avea o valoare exacta si comparam
        # cu media de pe celelalte luni si daca este in crestere atunci afisam ce produse contine preparatul
        # pentru a stii ce sa comande pentru stoc pe viitor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    firebase_initialize()
    db = firebase_connection()
    result_data = []
    # extragem toate produsele din firestore si facem o predictie pentru fiecare preparat
    data = db.collection(u'products').stream()
    for product in data:
        #extragem json-ul aferent {food, ingredients, sellPrice} fiecarui preparat
        product_container = product.to_dict()
        xgboost_regression(product_container['food'])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
