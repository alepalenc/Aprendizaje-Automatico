# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Alejandro Palencia Blanco
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse


np.random.seed(1)



print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1.2\n')


def E(u,v):
    return (u**3*np.exp(v-2)-2*v**2*np.exp(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(3*u**2*np.exp(v-2)+2*v**2*np.exp(-u))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(u**3*np.exp(v-2)-4*v*np.exp(-u))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(E, gradE, w0, lr, error, max_iter, verbose=False):
    w = np.copy(w0)
    w.astype(np.float64)
    iterations = 0
    values = np.array([E(w[0],w[1])])
    while E(w[0],w[1]) >= error and iterations < max_iter:
        w -= lr*gradE(w[0],w[1])
        iterations += 1
        values = np.append(values,E(w[0],w[1]))
        if verbose:
            print("iter:",iterations,"   w:",w,"   E(w)",E(w[0],w[1]))
    return w, iterations, values




eta = 0.1
max_iter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it, values = gradient_descent(E, gradE, initial_point, eta, error2get, max_iter)


print ('Número de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


# DISPLAY FIGURE
x = np.linspace(0.5, 1.5, 50)
y = np.linspace(0.5, 1.5, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()




input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 1.3\n')




def F(x,y):
    return (x+2)**2 + 2*(y-2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

#Derivada parcial de F con respecto a x
def dFx(x,y):
    return 2*(x+2) + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    
#Derivada parcial de F con respecto a y
def dFy(x,y):
    return 4*(y-2) + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

#Gradiente de F
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])


max_iter = 50
error2get = -2


eta = 0.01
initial_point = np.array([-1.0,1.0])
w, it, values = gradient_descent(F, gradF, initial_point, eta, error2get, max_iter)


print ('Tasa de aprendizaje: ', eta)
print ('Número de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
x = np.linspace(-4.0, 1.0, 50)
y = np.linspace(-1.0, 4.0, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y) #F_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente con lr=0.01')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



print("Evolución de los valores devueltos por los mínimos obtenidos en cada iteración para eta=0.01")
plt.scatter(np.arange(0,it+1), values, s=10)
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



eta = 0.1
initial_point = np.array([-1.0,1.0])
w, it, values = gradient_descent(F, gradF, initial_point, eta, error2get, max_iter)

print ('Tasa de aprendizaje: ', eta)
print ('Número de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
x = np.linspace(-4.0, 1.0, 50)
y = np.linspace(1.5, 4.0, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y) #F_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente con lr=0.1')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



print("Evolución de los valores devueltos por los mínimos obtenidos en cada iteración para eta=0.1")
plt.scatter(np.arange(0,it+1), values, s=10)
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



eta = 0.01

initial_point = np.array([-0.5,-0.5])
w, it, values = gradient_descent(F, gradF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([1.5,1.5])
w, it, values = gradient_descent(F, gradF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([2.1,-2.1])
w, it, values = gradient_descent(F, gradF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([-3.0,3.0])
w, it, values = gradient_descent(F, gradF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([-2.0,2.0])
w, it, values = gradient_descent(F, gradF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])



input("\n--- Pulsar tecla para continuar ---\n")












###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2.1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return np.mean((np.matmul(x,w)-y)**2)

# Gradiente Descendente Estocastico
def sgd(Err, x, y, w0, lr, error, batch_size, max_epochs):
    w = np.copy(w0)
    w.astype(np.float64)
    epoch = 1
    while Err(x,y,w) >= error and epoch <= max_epochs:
        minibatches = np.split(np.random.permutation(np.arange(y.size)), 
                               range(batch_size,y.size,batch_size))
        for mb in minibatches:
            w_aux = np.copy(w)
            for k in range(w.size):
                w[k] -= lr*(2/mb.size)*np.dot(np.matmul(x[mb,:],w_aux)-y[mb], x[mb,k])
        epoch += 1
    return w

# Pseudoinversa	
def pseudoinverse(x, y):
    return np.matmul(np.linalg.pinv(x),y)


# Lectura de los datos de entrenamiento
x_train, y_train = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


eta = 0.01
max_epochs = 50
batch_size = 32
error2get = 1e-3
initial_point = np.array([0.0,0.0,0.0])
w_sgd = sgd(Err, x_train, y_train, initial_point, eta, error2get, max_epochs, batch_size)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x_train, y_train, w_sgd))
print ("Eout: ", Err(x_test, y_test, w_sgd))



input("\n--- Pulsar tecla para continuar ---\n")



w_pinv = pseudoinverse(x_train, y_train)
print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x_train, y_train, w_pinv))
print ("Eout: ", Err(x_test, y_test, w_pinv))



input("\n--- Pulsar tecla para continuar ---\n")




# Función para calcular y en la recta w[0] + x*w[1] + y*w[2] = 0 para un x dado
def y_line(x,w):
    return -(w[0]+x*w[1])/w[2]


# Función para calcular el error de clasificación
def ErrClasif_y_line(x,y,w,verbose=False):
    prop_label1 = np.zeros(2)
    prop_label1neg = np.zeros(2)
    
    for i in range(y.size):
        if y[i] == 1:
            if x[i][2] - y_line(x[i][1],w) > 0:
                prop_label1[0] += 1
            else:
                prop_label1[1] += 1
        elif y[i] == -1:
            if x[i][2] - y_line(x[i][1],w) > 0:
                prop_label1neg[0] += 1
            else:
                prop_label1neg[1] += 1
    
    error_label1 = float(prop_label1[0])/float(prop_label1[0]+prop_label1[1])
    error_label1neg = float(prop_label1neg[1])/float(prop_label1neg[0]+prop_label1neg[1])
    error = float(prop_label1[0]+prop_label1neg[1])/float(prop_label1.sum()+prop_label1neg.sum())
    
    if verbose:
        print('Error asociado a la etiqueta 1: %f %%, %d de %d'
              % (error_label1*100, prop_label1[0], prop_label1[0]+prop_label1[1]))
        print('Error asociada a la etiqueta -1: %f %%, %d de %d'
              % (error_label1neg*100, prop_label1neg[1], prop_label1neg[0]+prop_label1neg[1]))
        print('Error de clasificación: %f %%, %d de %d'
              % (error*100, prop_label1[0]+prop_label1neg[1], prop_label1.sum()+prop_label1neg.sum()))
    
    return error


print("Gráfica comparativa solución SGD vs Pseudoinversa sobre datos de train")

plt.plot((0.0,1.0), (y_line(0.0,w_sgd),y_line(0.6,w_sgd)), c='green', label='SGD')
plt.plot((0.0,1.0), (y_line(0.0,w_pinv),y_line(0.6,w_pinv)), c='orange', label='pseudoinversa')
    
x_train_label5 = x_train[np.where(y_train == label5)]
plt.scatter(x_train_label5[:,1], x_train_label5[:,2], c="red", s=10, label='5')
x_train_label1 = x_train[np.where(y_train == label1)]
plt.scatter(x_train_label1[:,1], x_train_label1[:,2], c="blue", s=10, label='1')

plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.title('Solución SGD vs Pseudoinversa sobre datos de train')
plt.legend()
plt.show()


print('\nSGD train:')
ErrClasif_y_line(x_train, y_train, w_sgd, True)

print('\nPseudoinversa train:')
ErrClasif_y_line(x_train, y_train, w_pinv, True)




input("\n--- Pulsar tecla para continuar ---\n")




print("Gráfica comparativa solución SGD vs Pseudoinversa sobre datos de test")

plt.plot((0.0,1.0), (y_line(0.0,w_sgd),y_line(0.6,w_sgd)), c='green', label='SGD')
plt.plot((0.0,1.0), (y_line(0.0,w_pinv),y_line(0.6,w_pinv)), c='orange', label='pseudoinversa')
    
x_test_label5 = x_test[np.where(y_test == label5)]
plt.scatter(x_test_label5[:,1], x_test_label5[:,2], c="red", s=10, label='5')
x_test_label1 = x_test[np.where(y_test == label1)]
plt.scatter(x_test_label1[:,1], x_test_label1[:,2], c="blue", s=10, label='1')

plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.title('Solución SGD vs Pseudoinversa sobre datos de test')
plt.legend()
plt.show()


print('\nSGD test:')
ErrClasif_y_line(x_test, y_test, w_sgd, True)

print('\nPseudoinversa test:')
ErrClasif_y_line(x_test, y_test, w_pinv, True)




input("\n--- Pulsar tecla para continuar ---\n")









print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 



sample = simula_unif(1000, 2, 1.0)


print("Mapa de puntos 2D sin etiquetar")

plt.scatter(sample[:,0], sample[:,1], c="red", s=10)
plt.title('Mapa de puntos 2D sin etiquetar')
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")




labels = np.array([f(sample[i][0],sample[i][1]) for i in range(sample.shape[0])])
labels = labels * np.random.choice([1.0,-1.0], labels.size, p=[0.9,0.1])


print("Mapa de puntos 2D etiquetados")

sample1 = sample[np.where(labels == 1)]
plt.scatter(sample1[:,0], sample1[:,1], c="red", s=10, label='1')
sample1neg = sample[np.where(labels == -1)]
plt.scatter(sample1neg[:,0], sample1neg[:,1], c="blue", s=10, label='-1')
plt.title('Mapa de puntos 2D etiquetados')
plt.show()




input("\n--- Pulsar tecla para continuar ---\n")




# Función para calcular x en la recta w[0] + x*w[1] + y*w[2] = 0 para un y dado
def x_line(y,w):
    return -(w[0]+y*w[2])/w[1]


# Función para calcular el error de clasificación
def ErrClasif_x_line(x,y,w,verbose=False):
    prop_label1 = np.zeros(2)
    prop_label1neg = np.zeros(2)
    
    for i in range(y.size):
        if y[i] == 1:
            if x[i][1] - x_line(x[i][2],w) > 0:
                prop_label1[0] += 1
            else:
                prop_label1[1] += 1
        elif y[i] == -1:
            if x[i][1] - x_line(x[i][2],w) > 0:
                prop_label1neg[0] += 1
            else:
                prop_label1neg[1] += 1
    
    error_label1 = float(prop_label1[0])/float(prop_label1[0]+prop_label1[1])
    error_label1neg = float(prop_label1neg[1])/float(prop_label1neg[0]+prop_label1neg[1])
    error = float(prop_label1[0]+prop_label1neg[1])/float(prop_label1.sum()+prop_label1neg.sum())
    
    if verbose:
        print('Error asociado a la etiqueta 1: %f %%, %d de %d'
              % (error_label1*100, prop_label1[0], prop_label1[0]+prop_label1[1]))
        print('Error asociada a la etiqueta -1: %f %%, %d de %d'
              % (error_label1neg*100, prop_label1neg[1], prop_label1neg[0]+prop_label1neg[1]))
        print('Error de clasificación: %f %%, %d de %d'
              % (error*100, prop_label1[0]+prop_label1neg[1], prop_label1.sum()+prop_label1neg.sum()))
    
    return error


x_train = np.concatenate((np.ones((sample.shape[0],1),np.float64),sample), axis=1)
y_train = labels

eta = 0.05
batch_size = 16
max_epochs = 10
error2get = 1e-3
initial_point = np.array([0.0,0.0,0.0])

w = sgd(Err, x_train, y_train, initial_point, eta, error2get, batch_size, max_epochs)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x_train, y_train, w))
ErrClasif_x_line(x_train, y_train, w, True)


print("Gráfica solución SGD")

plt.plot((x_line(-1.0,w),x_line(1.0,w)), (-1.0,1.0), c='green', label='SGD')
    
x_train_label1 = x_train[np.where(y_train == 1)]
plt.scatter(x_train_label1[:,1], x_train_label1[:,2], c="red", s=10, label='1')
x_train_label1neg = x_train[np.where(y_train == -1)]
plt.scatter(x_train_label1neg[:,1], x_train_label1neg[:,2], c="blue", s=10, label='-1')

plt.title('Solución SGD')
plt.legend()
plt.show()






input("\n--- Pulsar tecla para continuar ---\n")



# Función para generar una muestra etiquetada
def generateLabeledSample(N, d, size):
    x = simula_unif(N, d, size)
    x = np.concatenate((np.ones((x.shape[0],1),np.float64),x), axis=1)
    
    y = np.array([f(x[i][1],x[i][2]) for i in range(x.shape[0])])
    y = y * np.random.choice([1.0,-1.0], y.size, p=[0.9,0.1])
    
    return x, y


eta = 0.05
batch_size = 16
max_epochs = 10
error2get = 1e-3
initial_point = np.array([0.0,0.0,0.0])

list_ein = []
list_eout = []
list_ein_clasif = []
list_eout_clasif = []


for i in range(1000):
    x_train, y_train = generateLabeledSample(1000, 2, 1.0)
    
    w = sgd(Err, x_train, y_train, initial_point, eta, error2get, batch_size, max_epochs)
    list_ein.append(Err(x_train, y_train, w))
    list_ein_clasif.append(ErrClasif_x_line(x_train, y_train, w))
    
    x_test, y_test = generateLabeledSample(1000, 2, 1.0)
    
    list_eout.append(Err(x_test, y_test, w))
    list_eout_clasif.append(ErrClasif_x_line(x_test, y_test, w))


list_ein = np.array(list_ein)
print("Valor medio Ein:", np.mean(list_ein))

list_ein_clasif = np.array(list_ein_clasif)
print("Valor medio Ein Clasificación:", 100*np.mean(list_ein_clasif), "%")

list_eout = np.array(list_eout)
print("Valor medio Eout:", 100*np.mean(list_eout))

list_ein_clasif = np.array(list_ein_clasif)
print("Valor medio Eout Clasificación:", 100*np.mean(list_eout_clasif), "%")



input("\n--- Pulsar tecla para continuar ---\n")



# Función para generar una muestra no etiquetada con características no lineales
def generateNonLinearLabeledSample(N, d, size):
    x, y = generateLabeledSample(N, d, size)
    x = np.concatenate(( x, np.reshape(x[:,1]*x[:,2],(-1,1)) ), axis=1)
    x = np.concatenate(( x, np.reshape(x[:,1]**2,(-1,1)) ), axis=1)
    x = np.concatenate(( x, np.reshape(x[:,2]**2,(-1,1)) ), axis=1)
    
    return x, y


# Función para calcular el error de clasificación
def ErrClasif_ellipse(x,y,w,verbose=False):
    prop_label1 = np.zeros(2)
    prop_label1neg = np.zeros(2)
    
    for i in range(y.size):
        aux = np.dot(x[i],w)
        if y[i] == 1:
            if aux < 0:
                prop_label1[0] += 1
            else:
                prop_label1[1] += 1
        elif y[i] == -1:
            if aux < 0:
                prop_label1neg[0] += 1
            else:
                prop_label1neg[1] += 1
    
    error_label1 = float(prop_label1[0])/float(prop_label1[0]+prop_label1[1])
    error_label1neg = float(prop_label1neg[1])/float(prop_label1neg[0]+prop_label1neg[1])
    error = float(prop_label1[0]+prop_label1neg[1])/float(prop_label1.sum()+prop_label1neg.sum())
    
    if verbose:
        print('Error asociado a la etiqueta 1: %f %%, %d de %d'
              % (error_label1*100, prop_label1[0], prop_label1[0]+prop_label1[1]))
        print('Error asociada a la etiqueta -1: %f %%, %d de %d'
              % (error_label1neg*100, prop_label1neg[1], prop_label1neg[0]+prop_label1neg[1]))
        print('Error de clasificación: %f %%, %d de %d'
              % (error*100, prop_label1[0]+prop_label1neg[1], prop_label1.sum()+prop_label1neg.sum()))
    
    return error



eta = 0.05
batch_size = 16
max_epochs = 10
error2get = 1e-3
initial_point = np.zeros(6)

x_train, y_train = generateNonLinearLabeledSample(1000, 2, 1.0)

w = sgd(Err, x_train, y_train, initial_point, eta, error2get, batch_size, max_epochs)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x_train, y_train, w))
ErrClasif_ellipse(x_train, y_train, w, True)


print("Gráfica solución SGD")

# Cálculo de los parámetros de la elipse: centro, ejes x e y, ángulo de rotación
denom = w[3]**2-4*w[4]*w[5]
root = np.sqrt((w[4]-w[5])**2+w[3]**2)
centre = ((2*w[5]*w[1]-w[3]*w[2])/denom, 
          (2*w[4]*w[2]-w[3]*w[1])/denom)
x_axe = -2*np.sqrt(2*(w[4]*w[2]**2+w[5]*w[1]**2-w[3]*w[1]*w[2]+denom*w[0])*(w[4]+w[5]+root))/denom
y_axe = -2*np.sqrt(2*(w[4]*w[2]**2+w[5]*w[1]**2-w[3]*w[1]*w[2]+denom*w[0])*(w[4]+w[5]-root))/denom
rot_angle = 0.0
if w[3] != 0:
    rot_angle = np.arctan2(w[5]-w[4]-root, w[3])
elif w[4] > w[5]:
    rot_angle = np.pi/2

plt.figure()
ax = plt.gca()
ellipse = Ellipse(xy=centre, width=x_axe, height=y_axe, angle=rot_angle, 
                  edgecolor='green', fc='None', lw=1)
ax.add_patch(ellipse)

x_train_label1 = x_train[np.where(y_train == 1)]
ax.scatter(x_train_label1[:,1], x_train_label1[:,2], c="red", s=10, label='1')
x_train_label1neg = x_train[np.where(y_train == -1)]
ax.scatter(x_train_label1neg[:,1], x_train_label1neg[:,2], c="blue", s=10, label='-1')

plt.title('Solución SGD')
plt.legend()
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



eta = 0.03
batch_size = 16
max_epochs = 10
error2get = 1e-3
initial_point = np.zeros(6)

list_ein = []
list_eout = []
list_ein_clasif = []
list_eout_clasif = []


for i in range(1000):
    x_train, y_train = generateNonLinearLabeledSample(1000, 2, 1.0)
    
    w = sgd(Err, x_train, y_train, initial_point, eta, error2get, batch_size, max_epochs)
    list_ein.append(Err(x_train, y_train, w))
    list_ein_clasif.append(ErrClasif_ellipse(x_train, y_train, w))
    
    x_test, y_test = generateNonLinearLabeledSample(1000, 2, 1.0)
    
    list_eout.append(Err(x_test, y_test, w))
    list_eout_clasif.append(ErrClasif_ellipse(x_test, y_test, w))


list_ein = np.array(list_ein)
print("Valor medio Ein:", np.mean(list_ein))

list_ein_clasif = np.array(list_ein_clasif)
print("Valor medio Ein Clasificación:", 100*np.mean(list_ein_clasif), "%")

list_eout = np.array(list_eout)
print("Valor medio Eout:", 100*np.mean(list_eout))

list_ein_clasif = np.array(list_ein_clasif)
print("Valor medio Eout Clasificación:", 100*np.mean(list_eout_clasif), "%")



input("\n--- Pulsar tecla para continuar ---\n")











###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('BONUS: MÉTODO DE NEWTON\n')


# Hessiana de f
def hessF(x,y):
    return np.array([[2-8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y), 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)],
                     [8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y), 4-8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)]])

# Método de Newton
def newton(E, gradE, hessE, w0, lr, error, max_iter, verbose=False):
    w = np.copy(w0)
    w.astype(np.float64)
    iterations = 0
    values = np.array([E(w[0],w[1])])
    while np.linalg.norm(gradE(w[0],w[1])) >= error and iterations < max_iter:
        w -= lr*np.dot(np.linalg.inv(hessE(w[0],w[1])),gradE(w[0],w[1]))
        iterations += 1
        values = np.append(values,E(w[0],w[1]))
        if verbose:
            print("iter:",iterations,"   w:",w,"   E(w)",E(w[0],w[1]))
    return w, iterations, values




max_iter = 50
error2get = 0


eta = 0.01
initial_point = np.array([-1.0,1.0])
w, it, values = newton(F, gradF, hessF, initial_point, eta, error2get, max_iter)


print ('Tasa de aprendizaje: ', eta)
print ('Número de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
x = np.linspace(-4.0, -1.0, 50)
y = np.linspace(1.5, 4.0, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y) #F_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=20)
ax.set(title='Bonus. Función sobre la que se calcula el método de Newton con lr=0.01')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



print("Evolución de los valores devueltos por los mínimos obtenidos en cada iteración para eta=0.01")
plt.scatter(np.arange(0,it+1), values, s=10)
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



eta = 0.1
initial_point = np.array([-1.0,1.0])
w, it, values = newton(F, gradF, hessF, initial_point, eta, error2get, max_iter)

print ('Tasa de aprendizaje: ', eta)
print ('Número de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
x = np.linspace(-4.0, 1.0, 50)
y = np.linspace(1.5, 4.0, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y) #F_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=20)
ax.set(title='Bonus. Función sobre la que se calcula el método de Newton con lr=0.1')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



print("Evolución de los valores devueltos por los mínimos obtenidos en cada iteración para eta=0.1")
plt.scatter(np.arange(0,it+1), values, s=10)
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



eta = 0.01

initial_point = np.array([-0.5,-0.5])
w, it, values = newton(F, gradF, hessF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([1.5,1.5])
w, it, values = newton(F, gradF, hessF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([2.1,-2.1])
w, it, values = newton(F, gradF, hessF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([-3.0,3.0])
w, it, values = newton(F, gradF, hessF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])

initial_point = np.array([-2.0,2.0])
w, it, values = newton(F, gradF, hessF, initial_point, eta, error2get, max_iter)
print("punto inicial:",initial_point,"   mínimo:",w,"   valor mínimo:",values[-1])



input("\n--- Pulsar tecla para continuar ---\n")



#Hessiana de E
def hessE(u,v):
    return np.array([[30*u**4*(np.exp(v-2))**2+16*v**4*(np.exp(-u))**2+4*(-u**2+6*u-6)*v**2*np.exp(-u)*np.exp(v-2), 
                      12*u**5*(np.exp(v-2))**2-32*v**3*np.exp(-u)**2+4*u**2*v*np.exp(-u)*np.exp(v-2)*(-6-3*v+2*u+u*v)],
                     [12*u**5*(np.exp(v-2))**2-32*v**3*np.exp(-u)**2+4*u**2*v*np.exp(-u)*np.exp(v-2)*(-6-3*v+2*u+u*v), 
                      4*u**6*(np.exp(v-2))**2+48*v**2*(np.exp(-u))**2+4*(-2-4*v-v**2)*u**3*np.exp(-u)*np.exp(v-2)]])


eta = 0.1
max_iter = 10
error2get = 0
initial_point = np.array([1.0,1.0])


w, it, values = gradient_descent(E, gradE, initial_point, eta, error2get, max_iter)

print ('Gradiente descendente sobre la función E:')
print ('Número de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Valor mínimo:', values[-1])
print("Evolución de los valores devueltos por los mínimos obtenidos en cada iteración por gradiente descendente")
plt.scatter(np.arange(0,it+1), values, s=4)
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")



eta = 1
max_iter = 7

w, it, values = newton(E, gradE, hessE, initial_point, eta, error2get, max_iter)

print ('Método de Newton sobre la función E:')
print ('Número de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Valor mínimo:', values[-1])
print("Evolución de los valores devueltos por los mínimos obtenidos en cada iteración por el método de Newton")
plt.scatter(np.arange(0,it+1), values, s=4)
plt.show()
