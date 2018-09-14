from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
Se utilizara la red neuronal programada en clase con un dataset diferente, en este caso el dataset es de factores 
que se pueden tomar en cuenta para predecir el indice de masa corporal de una persona

'''
dataset = pd.read_csv("./datosHospital.csv")
#Impresion de columnas
print(dataset.columns)
# Se seleccionaron las columnas más relevantes según la investigación para determinar el indice de masa corporal
selected_columns = ['peso', 'estatura', 'Cintura', 'colest', 'trigl', 'HDL','edad','Ejercic','Aliment']
# el objetivo es predecir la columna de imc
target = ['imc_']
# Se realiza el corte del dataset definiendo x y y
x = dataset.loc[:, selected_columns]
y = dataset.loc[:, target]
# Se decidio realizar una división de 80% de datos para entrenamiento y 20% de datos para prueba
xtrain = x[1:80]
xtest = x[80:]
ytrain = y[1:80]
ytest = y[80:]
# Se realizo la normalización de los datos
xtrain = xtrain / np.amax(xtrain, axis=0)
ytrain = ytrain / np.amax(ytrain, axis=0)
xtest = xtest / np.amax(xtest, axis=0)
ytest = ytest / np.amax(ytest, axis=0)
#Convertir el dataset a numpy array
xtraina = xtrain.values
xtesta = xtest.values
ytraina = ytrain.values
ytesta = ytest.values

class RedNeuronal(object):
    def __init__(self, Lambda = 0):
        self.inputs = 9
        self.outputs = 1
        '''
        Se realizaron varias pruebas y se determino que 80 daba una de las predicciones
        más precisas del imc al graficar Costo vs Iteracion
        '''
        self.hidden = 80
        self.W1 = np.random.randn(self.inputs, self.hidden)
        self.W2 = np.random.randn(self.hidden, self.outputs)
        self.Lambda = Lambda
    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))

    def feedForward(self, x):
        self.z2 = x @ self.W1
        self.a2 = self.sigmoide(self.z2)
        self.z3 = self.a2 @ self.W2
        self.yhat = self.sigmoide(self.z3)
        return self.yhat

    def sigmoideDerivada(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def funcionCosto(self, x, y):
        self.yhat = self.feedForward(x)
        Costo = 0.5 * sum((y - self.yhat) ** 2)/x.shape[0] +(self.Lambda/2) * (np.sum(self.W1**2)+ np.sum(self.W2**2))
        return Costo

    def funcionDeCostoDerivada(self, x, y):
        self.yhat = self.feedForward(x)
        self.delta3 = np.multiply(-(y - self.yhat), self.sigmoideDerivada(self.z3))
        djW2 = (np.transpose(self.a2) @ self.delta3)/x.shape[0]+(self.Lambda * self.W2)
        self.delta2 = self.delta3 @ djW2.T * self.sigmoideDerivada(self.z2)
        djW1 = (x.T @ self.delta2)/x.shape[0]+(self.Lambda * self.W1)
        return djW1, djW2

    def getPesos(self):
        data = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return data

    def setPesos(self, datos):
        W1_inicio = 0
        W1_fin = self.hidden * self.inputs
        self.W1 = np.reshape(datos[W1_inicio:W1_fin], (self.inputs, self.hidden))
        W2_fin = W1_fin + self.hidden * self.outputs
        self.W2 = np.reshape(datos[W1_fin:W2_fin], (self.hidden, self.outputs))

    def getGradientes(self, X, y):
        djW1, djW2 = self.funcionDeCostoDerivada(X, y)
        return np.concatenate((djW1.ravel(), djW2.ravel()))




class Entrenador:
    def __init__(self, unaRed):
        # referencia a una red local
        self.NN = unaRed

    def actualizaPesos(self, params):
        self.NN.setPesos(params)
        self.Costos.append(self.NN.funcionCosto(self.X, self.y))
        self.CostosTest.append( \
            self.NN.funcionCosto(self.Xtest, self.ytest))

    def obtenPesosNN(self, params, X, y):
        self.NN.setPesos(params)
        cost = self.NN.funcionCosto(X, y)
        grad = self.NN.getGradientes(X, y)
        return cost, grad

    def entrena(self, Xtrain, ytrain, Xtest, ytest):
        # variables para funciones callback
        self.X = Xtrain
        self.y = ytrain

        self.Xtest = Xtest
        self.ytest = ytest

        # lista temporal de costos
        self.Costos = []
        self.CostosTest = []

        pesos = self.NN.getPesos()

        opciones = {'maxiter': 300, 'disp': True}

        # self.obtenPesosNN, funcion objetivo
        # args=(X, y), input / output data
        # salida, regresa el costo y los gradientes
        salida = optimize.minimize(self.obtenPesosNN, pesos, jac=True, method='BFGS', \
                                   args=(Xtrain, ytrain), \
                                   options=opciones, \
                                   callback=self.actualizaPesos)

        self.NN.setPesos(salida.x)
        self.resultados = salida

rn = RedNeuronal()
e = Entrenador(rn)
e.entrena(xtraina, ytraina, xtesta, ytesta)
plt.plot(e.Costos)
plt.plot(e.CostosTest)
plt.grid(1)
plt.ylabel("Costo")
plt.xlabel("Iteración")
plt.show()