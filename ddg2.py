import numpy as np

x= np.random.rand(100)
x_0=np.ones(len(x))
y=2+2*x
print("funcion real: y=2 + 2 x")

X=np.zeros((len(x),2))
for i in list(range(len(x))):
    X[i,0]=x_0[i]
    X[i,1]=x[i]

theta=np.random.rand(2)
h_theta=theta[0]+theta[1]*x
print("theta iniciales:",theta)

#DdG
alpha=0.00005


suma=np.dot((np.linalg.inv(np.dot(np.transpose(X),X))),np.transpose(X))
theta[0]=np.dot(suma[0,:],y)
theta[1]=np.dot(suma[1,:],y)


h_theta=theta[0]+theta[1]*x
loss=np.sum((h_theta-y)**2)/2
print("funcion de perdida final: ",loss,"\ntheta finales :", theta)
