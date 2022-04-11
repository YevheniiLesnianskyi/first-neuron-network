from math import exp
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class neuron:

    def __init__(self, n, bias, maks_error, sygnaly):
        self.n = n
        self.bias = bias
        self.maks_error = maks_error
        self.sygnaly = sygnaly

    def wagi(self, ilosc):
        self.zbior_wag = []
        for j in range(ilosc + 1):
            zmienna = []
            for i in range(ilosc + 1):
                zmienna.append(random.uniform(-1, 1))
            self.zbior_wag.append(zmienna)
        return self.zbior_wag

    # Obliczenie dla 1 neuronu
    def calculate(self, wagi, inputs, bias):
        calculation = 0
        for i in range(len(wagi) - 1):
            calculation += wagi[i] * inputs[i]
        calculation += wagi[-1] * bias
        return calculation

    def sigmoidalna(self, wagi, row, bias):
        activ_outputs = self.calculate(wagi, row, bias)
        active_transfer_outputs = 1 / (1 + exp(-activ_outputs))
        return active_transfer_outputs

    def obliczanie(self, wagi1, wagi2, wagi3, limit_iteracji):
        for a in range(1, limit_iteracji + 1):
            error = 0
            for wektor in self.sygnaly:
                wynik = wektor[-1]
                wektor = wektor[:-1]
                pierwsze_wyniki = self.sigmoidalna(wagi1, wektor, self.bias)
                drugie_wyniki = self.sigmoidalna(wagi2, wektor, self.bias)
                koncowe_wyniki = self.sigmoidalna(wagi3, [pierwsze_wyniki, drugie_wyniki], self.bias)
                # Uaktualniamy wagi
                delta, first_w, second_w = self.aktualizacja_wag_2_warstwa(wagi3, koncowe_wyniki, self.n, wynik, self.bias,
                                                                      [pierwsze_wyniki, drugie_wyniki])
                self.aktualizacja_wag_1_warstwa(wagi1, pierwsze_wyniki, self.n, self.bias, delta, first_w, wektor)
                self.aktualizacja_wag_1_warstwa(wagi2, drugie_wyniki, self.n, self.bias, delta, second_w, wektor)

                error += 0.5 * ((wynik - koncowe_wyniki) ** 2)

            if error <= self.maks_error:
                break

    # Uaktualnienie wag
    def aktualizacja_wag_1_warstwa(self, wagi, output, n, bias, delta, poprzednia_waga, inputs):
        delta = delta * poprzednia_waga * (output * (1.0 - output))
        for i in range(len(wagi) - 1):
            wagi[i] += n * delta * inputs[i]
        wagi[-1] += n * delta * bias


    # Uaktualnienie wag
    def aktualizacja_wag_2_warstwa(self, wagi, output, n, wynik, bias, inputs):
        delta = (wynik - output) * (output * (1.0 - output))
        for i in range(len(wagi) - 1):
            wagi[i] += n * delta * inputs[i]
        wagi[-1] += n * delta * bias
        return delta, wagi[0], wagi[1]


    def predict(self, wagi_dla_1neuro, wagi_dla_2neuro, wagi_dla_3neuro, wektor, bias):
        pierwsze_wyniki = self.sigmoidalna(wagi_dla_1neuro, wektor, bias)
        drugie_wyniki = self.sigmoidalna(wagi_dla_2neuro, wektor, bias)
        koncowe_wyniki = self.sigmoidalna(wagi_dla_3neuro, [pierwsze_wyniki, drugie_wyniki], bias)
        return (round(koncowe_wyniki), koncowe_wyniki)


sygnaly = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
obj = neuron(0.2, 1, 0.02,sygnaly)
zbior = obj.wagi(2)
print(zbior[0])
aasdasd = obj.obliczanie(zbior[0], zbior[1], zbior[2], 10000)



x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = np.zeros((100, 100))

# Dwie petli dla rysowania
for j in range(len(X1)):
    x1 = X1[j]
    x2 = X2[j]
    for i in range(len(x1)):
        row = [x1[i], x2[i]]
        prediction = obj.predict(zbior[0], zbior[1], zbior[2], row, 1)
        Y[j][i] = prediction[1]

fig = plt.figure(figsize=(10, 5))  # wielkosc okna
ax = plt.axes(projection='3d')  # 3d

surf = ax.plot_surface(X1, X2, Y, cmap=cm.viridis,
                       linewidth=0)
plt.show()
