import numpy as np

x= np.random.rand(35)
y=2+2*x 

theta=np.random.rand(2)
h_theta=theta[0]+theta[1]*x
print("theta iniciales:",theta)

#DdG
alpha=0.005

for i in list(range(10000)):
    suma_0=np.sum(h_theta-y)
    suma_1=np.dot(h_theta-y,x)

    theta[0]-=alpha*suma_0
    theta[1]-=alpha*suma_1
    h_theta=theta[0]+theta[1]*x
    loss=np.sum((h_theta-y)**2)/2
 
print("funcion de perdida final: ",loss,"\ntheta finales :", theta)
