#!/usr/bin/env python
# coding: utf-8

# # Creando red nueronal desde cero

# In[2]:


"""
importar las liberias para trabajar con los vectores nescsarios
asi como las librerias, que generen una los graficos de visualizacion
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles


# In[4]:


"""
El prolema que resuelve esta esta red neuronal
es un proceso de clasificación, en una forma
geometrica circular, que presenta un forma de clasificacion
en para la agrupacion de varios puntos caracteristicos 
"""
#crear el dataset
#numero de resgitros que se tiene en los datos
n = 500
#caractristicas que continen los datos
#ejemplo: peronas p(1)=edad, p(2)=altura
p = 2
"""
X, Y guardaran los datos de la funcion make_circles
donde n_samples=numero de circulos, factor=distancia entre los circulos
noise=ruido para la distorcion de plot 
"""
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
#print(Y)
Y = Y[:, np.newaxis]
"""
plt.scatter(X[Y[:, 0]==0,0], X[Y[:, 0]==0, 1], c="skyblue") 
plt.scatter(X[Y[:, 0]==1,0], X[Y[:, 0]==1, 1], c="salmon")
plt.axis("equal")
plt.show()
"""

# In[5]:


# clase de la ccapa de la red
class neural_layer():
    def __init__(self, n_conn, n_nuer, act_f):
        self.act_f = act_f 
        self.b = np.random.rand(1, n_nuer)      * 2-1
        self.W = np.random.rand(n_conn, n_nuer) * 2-1


# In[6]:


# funciones de activacion
# se definen funciones anonimas medinate lambda
sigm = (lambda x: 1 / ( 1 + np.e ** (-x)),
        lambda x: x * (1 - x))

relu = lambda x: np.maximum(0, x)
_x = np.linspace(-5 , 5, 100)
#plt.plot(_x, sigm[1](_x))
#plt.plot(_x, sigm[0](_x))
#plt.plot(_x, relu(_x))


# In[7]:


l0 = neural_layer(p, 4, sigm)
l1 = neural_layer(4, 8, sigm) 
#....
def create_nn(topology, act_f):
    nn = []
    for i, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[i], topology[i+1], act_f))
    return nn
#topology = [p, 4, 8, 16, 8, 4, 1 ]
#create_nn(topology, sigm)


# In[12]:


#topology = [p, 4, 8, 16, 8, 4, 1 ]
topology = [p, 4, 8, 1 ]

neural_net = create_nn(topology, sigm)
#funcion de predccion del error, mediante el error cuadratico medio
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))
#metodo para el resultado esperado, conforme a lr=nivel_o_radio_de_aprendizaje
def train(neural_net, X, Y, l2_cost, lr=0.5, train = True):
    #pasos hacia delante para el entrenamiento
    out = [(None, X)]#vector que guarda sumas ponderadas de los datos y la act_f
    for i, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[i].W + neural_net[i].b
        a = neural_net[i].act_f[0](z)
        
        out.append((z, a))
    #print(out[-1][1])    
    #print(l2_cost[0](out[-1][1], Y))
    ##
    if train:
        #pasos hacia atras
        deltas = []
        for i in reversed(range(0, len(neural_net))):
            z = out[i+1][0]
            a = out[i+1][1]
            
            
             #print(a.shape)
            
            if i == len(neural_net) - 1:
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[i].act_f[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[i].act_f[1](a))
            
            _W = neural_net[i].W
            
            #descenso del gradiente
            neural_net[i].b = neural_net[i].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            #print (out[i][1].shape, deltas[0].shape)
            neural_net[i].W = neural_net[i].W - out[i][1].T @ deltas[0] * lr
    
    return out[-1][1]
train(neural_net, X, Y, l2_cost, 0.5)
print("")


# In[16]:


import time 
from IPython.display import clear_output
neural_n = create_nn(topology, sigm)
perdida = []

for i in range (1000):
    #entrenamos a la red
    pY = train(neural_n, X, Y, l2_cost, lr=0.05)
    print(i)
    if i % 25 == 0:
        perdida.append(l2_cost[0](pY, Y))
        
        res = 50
        
        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
        
        _Y = np.zeros((res, res))
        
        
        for i0, x0 in enumerate(_x0): 
            for i1, x1 in enumerate (_x1):
                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train = False)[0][0]
        
        plt.pcolormesh(_x0, _x1, _Y,cmap="coolwarm")
        plt.axis("equal")
        
        plt.scatter(X[Y[:, 0]==0,0], X[Y[:, 0]==0, 1], c="skyblue") 
        plt.scatter(X[Y[:, 0]==1,0], X[Y[:, 0]==1, 1], c="salmon")
        
        clear_output(wait=True)
        plt.show()
        plt.plot(range(len(perdida)), perdida)
        plt.show()
        time.sleep(0.5)


# In[ ]:




