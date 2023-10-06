import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


@tf.function
def edo_func(modelo,t):
  x=modelo(t)
  dx=tf.gradients(x,t)[0]
  ddx=tf.gradients(dx,t)[0]
  y_nau=float(ddx)+1.0*x
  return y_nau

#funcion de coste
def perdida(modelo,t,t_i):
  y_nau=edo_func(modelo,t)
  y_inicial=cond_inicial(modelo,t_i)
  return tf.reduce_sum(tf.square(y_nau))+tf.reduce_sum(tf.square(y_inicial[0]))+tf.reduce_sum(tf.square(y_inicial[1]))

#gradietne de la perdida
def perdida_gradiente(modelo,t,t_i):
  with tf.GradientTape() as tape:
    loss=perdida(modelo,t,t_i)
  grand=tape.gradient(loss,modelo.trainable_weights)
  return [loss,grand]


@tf.function
def cond_inicial(modelo,t_i):

  x_0_mol=modelo(t_i)
  dx_0_mol=tf.gradients(x_0_mol,t_i)[0]

  x_0_real=0.0
  dx_0_real=1.0
  return [x_0_mol - x_0_real , dx_0_mol - dx_0_real]

def optimizar(modelo,t,t_i,N=1000):
  optimi=tf.keras.optimizers.Adam()
  err=[]
  for i in list(range(N)):
    grad=perdida_gradiente(modelo,t,t_i)
    loss=grad[0]
    value=grad[1]
    optimi.apply_gradients(zip(value,modelo.trainable_weights))
    if i%100==1:
      print("Entrenamiento :",i,"perdida :",float(loss))
    err.append(loss)
  return err

inputs=tf.keras.Input(shape=(1,))
capa1=tf.keras.layers.Dense(units=64,activation=tf.tanh)(inputs)
capa2=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa1)
capa3=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa2)
capa4=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa3)#units = cantidad de naurona
outputs=tf.keras.layers.Dense(units=1,activation=None)(capa4)#dense : neurona que esta conectada con todas las anteriores



t=np.linspace(0.0,2.0*np.pi,20)
t_col=t[:,None] 
#idx=[np.random.choice(t.shape[0],20, replace=False)]
#t_col=t_col[idx,:][0]


model2=tf.keras.Model(inputs,outputs)

opt=optimizar(model2,t_col,np.array([0.0,],dtype=np.float32)[:,None])



plt.plot(t,model2(t_col),"--r")
plt.plot(t,np.sin(t)*max(model2(t_col)),"--k")
plt.legend(("NN","seno(t)"))
plt.show()
plt.plot(list(range(len(opt))),opt)
plt.show()