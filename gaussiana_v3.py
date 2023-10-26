import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools as it
import tensorflow as tf
from matplotlib.animation import FuncAnimation
import time

def gauss(t,x,y,c,lam):
  centro=10.0


  r=tf.sqrt((x-centro)**2+(y-centro)**2)
  
  return 4.0*tf.exp(-(((r-c*t)/lam)**2)) 

@tf.function
def edo_func(modelo,t,x,y):
  feature=tf.concat([t,x,y],1)
  phi=modelo(feature)
  # x
  dphix=tf.gradients(phi,x)[0]
  dphiy=tf.gradients(phi,y)[0]
  dphit=tf.gradients(phi,t)[0]


  ddx_phi=tf.gradients(dphix,x)[0]
  ddy_phi=tf.gradients(dphiy,y)[0]
  ddt_phi=tf.gradients(dphit,t)[0]

  c=0.5


  zeta=ddt_phi-c*((ddx_phi)+(ddy_phi))
  return zeta

@tf.function
def condiciones(modelo,t_1,t_2,x_i,y_i,feature_r):
  feature_i1=tf.concat([t_1,x_i,y_i],1)
  feature_i2=tf.concat([t_2,x_i,y_i],1)

  t1_mol=modelo(feature_i1)
  t2_mol=modelo(feature_i2)

  dx_phi1=tf.gradients(t1_mol,x_i)[0]
  dy_phi1=tf.gradients(t1_mol,y_i)[0]
  dt_phi1=tf.gradients(t1_mol,t_1)[0]


  dx_phi2=tf.gradients(t2_mol,x_i)[0]
  dy_phi2=tf.gradients(t2_mol,y_i)[0]
  dt_phi2=tf.gradients(t2_mol,t_2)[0]


  t1=feature_r[:,0:1]
  t2=feature_r[:,1:2]

  dphi1_x=feature_r[:,2:3]
  dphi1_y=feature_r[:,3:4]
  dphi1_t=feature_r[:,4:5]

  dphi2_x=feature_r[:,5:6]
  dphi2_y=feature_r[:,6:7]
  dphi2_t=feature_r[:,7:8]

  err=tf.reduce_mean(tf.square([t2_mol-t2 , t1_mol-t1]))
  err+=tf.reduce_mean(tf.square([dx_phi1-dphi1_x, dy_phi1-dphi1_y, dt_phi1-dphi1_t 
                                 ,dx_phi2-dphi2_x , dy_phi2 - dphi2_y, dt_phi2- dphi2_t]))
  return err
#funcion de coste
def perdida(modelo,t,x,y,t_1,t_2,x_i,y_i,feature_r):
  zeta=edo_func(modelo,t,x,y)
  zeta_inicial=condiciones(modelo,t_1,t_2,x_i,y_i,feature_r)
  zeta_nau=tf.reduce_mean(tf.square(zeta))
  return zeta_nau+zeta_inicial,zeta_nau,zeta_inicial

#gradietne de la perdida
def perdida_gradiente(modelo,t,x,y,t_1,t_2,x_i,y_i,feature_r):
  with tf.GradientTape() as tape:
    loss=perdida(modelo,t,x,y,t_1,t_2,x_i,y_i,feature_r)
  grand=tape.gradient(loss[0],modelo.trainable_weights)
  return [loss,grand]




def optimizar(modelo,t,x,y,t_1,t_2,x_i,y_i,feature_r,N=40000):
  optimi=tf.keras.optimizers.Adam( learning_rate=0.0003)
  err_t=[];err_ede=[];err_con=[]
  for i in list(range(N)):
    grad=perdida_gradiente(modelo,t,x,y,t_1,t_2,x_i,y_i,feature_r)
    loss=grad[0]
    value=grad[1]
    optimi.apply_gradients(zip(value,modelo.trainable_weights))
    if i%100==0:
      print("Entrenamiento :",i,"perdida total :",float(loss[0])," ecuacion de onda :",float(loss[1]),
            " condiciones :",float(loss[2]))
    err_t.append(loss[0])
    err_ede.append(loss[1])
    err_con.append(loss[2])

  return err_t,err_ede,err_con



inicio = time.time()



L=20

x=np.linspace(0.0,20.0,L)
y=np.linspace(0.0,20.0,L)
t=np.linspace(0.0,20.0,L)

var=list(it.product(t,x,y))

t_col=np.array(var,dtype=np.float32)[:,0][:,None]
x_col=np.array(var,dtype=np.float32)[:,1][:,None]
y_col=np.array(var,dtype=np.float32)[:,2][:,None]


t1=np.array(np.full((L*L,),[t[0],]),dtype=np.float32)[:,None]
t2=np.array(np.full((L*L,),[t[1],]),dtype=np.float32)[:,None]
xy=np.array(list(it.product(x,y)),dtype=np.float32)
#t1_i=np.array(list(it.product(t1,x,y)),dtype=np.float32)
#t2_i=np.array(list(it.product(t2,x,y)),dtype=np.float32)[:,0][:,None]


c=0.5
lam=1.0
phi_t1=gauss(t_col[0:400],x_col[0:400],y_col[0:400],c,lam)
phi_t2=gauss(t_col[400:800],x_col[400:800],y_col[400:800],c,lam)
x_tf=tf.Variable(x_col)
y_tf=tf.Variable(y_col)
t_tf=tf.Variable(t_col)

with tf.GradientTape(persistent=True) as tape:
    dphi = gauss(t_tf,x_tf,y_tf,c,lam)
dphi_x= tape.gradient(dphi, x_tf)
dphi_y= tape.gradient(dphi, y_tf)
dphi_t= tape.gradient(dphi, t_tf)

feature=np.concatenate((t_col,x_col,y_col),axis=1,dtype=np.float32)
feature_i1=np.concatenate((t1,xy),axis=1,dtype=np.float32)
feature_i2=np.concatenate((t2,xy),axis=1,dtype=np.float32)

feature_r=np.concatenate((phi_t1,phi_t2,dphi_x[0:400],dphi_y[0:400],dphi_t[0:400],dphi_x[400:800],dphi_y[400:800],dphi_t[400:800]),axis=1,dtype=np.float32)

inputs=tf.keras.Input(shape=[3,], dtype=tf.float32)
capa1=tf.keras.layers.Dense(units=64,activation=tf.tanh)(inputs)
capa2=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa1)
capa3=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa2)
capa4=tf.keras.layers.Dense(units=64,activation=tf.tanh)(capa3)

outputs=tf.keras.layers.Dense(units=1,activation=None)(capa4)


model=tf.keras.Model(inputs,outputs)
opt=optimizar(model,t_col,x_col,y_col,t1,t2,xy[:,0][:,None],xy[:,1][:,None],feature_r)


fin = time.time()
print("Tiempo del entrenamiento",fin-inicio,"segundos")
plt.plot(list(range(len(opt[0]))),opt[0])
plt.plot(list(range(len(opt[1]))),opt[1])
plt.plot(list(range(len(opt[2]))),opt[2])

plt.show()


dats=model(feature).numpy().reshape((L,L,L))

xx,yy=np.meshgrid(x,y)
plt.style.use('dark_background')
fig, axl = plt.subplots(subplot_kw=dict(projection='3d'))
#line, = axl.plot(X,data[0,:],"-k" )

def actualizarL(i):
    axl.clear()
    axl.set_zlim(-5, 5)
    axl.plot_surface(xx, yy, dats[i,:,:], cmap=cm.coolwarm,linewidth=0, antialiased=False)


chu=FuncAnimation(fig,actualizarL)
chu.save('gaussiana_tf10.gif')

#plt.show()
