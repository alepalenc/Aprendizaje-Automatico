# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Alejandro Palencia Blanco
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gauss(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x_unif = simula_unif(50, 2, [-50,50])

plt.scatter(x_unif[:,0], x_unif[:,1])
plt.title("Nube de puntos uniforme")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

x_gauss = simula_gauss(50, 2, np.array([5,7]))

plt.scatter(x_gauss[:,0], x_gauss[:,1])
plt.title("Nube de puntos gaussiana")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

# Función que dibuja una nube de puntos etiquetada junto con una recta en los límites especificados
def dibujar_nube_etiq_con_recta(x, y, a, b, etiq1, etiq1neg, x_inf, x_sup, y_inf, y_sup, title=''):
    x_label1 = x[np.where(y == 1)]
    plt.scatter(x_label1[:,0], x_label1[:,1], c="blue", s=5, label=etiq1)
    x_label1neg = x[np.where(y == -1)]
    plt.scatter(x_label1neg[:,0], x_label1neg[:,1], c="red", s=5, label=etiq1neg)
    t = np.linspace(x_inf,x_sup,2)
    plt.plot(t, a*t+b, c="green")
    plt.ylim(y_inf,y_sup)
    plt.title(title)
    plt.legend()
    plt.show()


a, b = simula_recta([-50,50])
x = simula_unif(100, 2, [-50,50])
y = np.array([f(x[i][0],x[i][1],a,b) for i in range(x.shape[0])])

dibujar_nube_etiq_con_recta(x, y, a, b, 1, -1, -50, 50, -50, 50, 
                            "Nube de puntos uniforme etiquetada")


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

# Para cada etiqueta, tomo los índices de los elementos con esa etiqueta,
#   los barajo y cambio el signo de la primera décima parte de ellos
y_ruido = np.copy(y)
for label in (1,-1):
    elem_label = np.where(y_ruido == label)[0]
    for elem in np.random.permutation(elem_label)[0:elem_label.size//10+1]:
        y_ruido[elem] *= -1

dibujar_nube_etiq_con_recta(x, y_ruido, a, b, 1, -1, -50, 50, -50, 50, 
                            "Nube de puntos uniforme etiquetada con un 10% de ruido")


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=1, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


def evaluate_grid(f, grid):
    z = np.zeros(len(grid))
    for i in range(len(grid)):
        z[i] = f(grid[i][0],grid[i][1])
    return z

def f1(x,y):
    return (x-10)**2 + (y-20)**2 - 400

def f2(x,y):
    return 0.5*(x+10)**2 + (y-20)**2 - 400

def f3(x,y):
    return 0.5*(x-10)**2 - (y+20)**2 - 400

def f4(x,y):
    return y - 20*x**2 - 5*x + 3

def f1_grid(grid):
    return evaluate_grid(f1, grid)

def f2_grid(grid):
    return evaluate_grid(f2, grid)

def f3_grid(grid):
    return evaluate_grid(f3, grid)

def f4_grid(grid):
    return evaluate_grid(f4, grid)


# Error de clasificación
def err_accuracy(x, y, f):
    acc = 0.0
    for i in range(y.size):
        if signo(f(x[i][0],x[i][1])) != y[i]:
            acc += 1
    return acc/y.size


plot_datos_cuad(x,y_ruido,f1_grid,'Clasificación realizada por f1 (circunferencia)')
print('Accuracy f1 (circunferencia):', err_accuracy(x,y,f1))

input("\n--- Pulsar tecla para continuar ---\n")

plot_datos_cuad(x,y_ruido,f2_grid,'Clasificación realizada por f2 (elipse)')
print('Accuracy f2 (elipse):', err_accuracy(x,y,f2))

input("\n--- Pulsar tecla para continuar ---\n")

plot_datos_cuad(x,y_ruido,f3_grid,'Clasificación realizada por f3 (hipérbola)')
print('Accuracy f3 (hipérbola):', err_accuracy(x,y,f3))

input("\n--- Pulsar tecla para continuar ---\n")

plot_datos_cuad(x,y_ruido,f4_grid,'Clasificación realizada por f4 (parábola)')
print('Accuracy f4 (parábola):', err_accuracy(x,y,f4))

input("\n--- Pulsar tecla para continuar ---\n")









###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

# Algoritmo Perceptrón
def ajusta_PLA(datos, label, max_iter, vini):
    w = np.copy(vini)
    w.astype(np.float64)
    iterations = 0
    change = True
    while change and iterations < max_iter:
        change = False
        i = 0
        while i < label.size and iterations < max_iter:
            if signo(np.dot(w,datos[i])) != label[i]:
                w += label[i]*datos[i]
                change = True
            i += 1
            iterations += 1
    return w, iterations

max_iter = 100000
datos = np.concatenate((np.ones((x.shape[0],1),np.float64),x), axis=1)
label = y

print('PERCEPTRÓN (sobre datos sin ruido)')

# Null vector initialization
w, iterations = ajusta_PLA(datos, label, max_iter, np.zeros(3))

print('Iteraciones necesarias para converger (vector inicial nulo): {}'.format(iterations))

# Random initializations
iterations = []
for i in range(0,10):
    vini = np.random.uniform(0,1,3)
    w, it = ajusta_PLA(datos, label, max_iter, vini)
    iterations.append(it)
    print("vini:",vini,"\titerations:",it)
    
print('Valor medio de iteraciones necesario para converger (vectores iniciales aleatorios): {}'.format(np.mean(np.asarray(iterations))))


input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b
print('PERCEPTRÓN (sobre datos con ruido)')

max_iter = 100000
datos = np.concatenate((np.ones((x.shape[0],1),np.float64),x), axis=1)
label = y_ruido

# Null vector initialization
w, iterations = ajusta_PLA(datos, label, max_iter, np.zeros(3))

print('Iteraciones necesarias para converger (vector inicial nulo): {}'.format(iterations))

# Random initializations
iterations = []
for i in range(0,10):
    vini = np.random.uniform(0,1,3)
    w, it = ajusta_PLA(datos, label, max_iter, vini)
    iterations.append(it)
    print("vini:",vini,"\titerations:",it)
    
print('Valor medio de iteraciones necesario para converger (vectores iniciales aleatorios): {}'.format(np.mean(np.asarray(iterations))))


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.2: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
print('REGRESIÓN LOGÍSTICA')

# Función que clasifica el item xi con la etiqueta 1 si P[yi=1/xi] >= 0.5
#    y con la etiqueta -1 en caso contrario
def clasificadorRL(w, xi):
    exp_wxi = np.exp(np.dot(w,xi))
    return signo(exp_wxi/(1+exp_wxi)-0.5)

# Algoritmo SGD adaptado a Regresión Logística
def sgdRL(x, y, w0, lr, error, batch_size, max_epochs):
    w = np.copy(w0)
    w.astype(np.float64)
    epoch = 0
    diff_w = error
    while diff_w >= error and epoch < max_epochs:
        epoch += 1
        w_prev = w.copy()
        minibatches = np.split(np.random.permutation(np.arange(y.size)), 
                               range(batch_size,y.size,batch_size))
        for mb in minibatches:
            grad = np.zeros(w.size, np.float64)
            for i in mb:
                grad = grad + (y[i]/(1+np.exp(y[i]*np.dot(w,x[i]))))*x[i]
            w = w + (lr/mb.size)*grad
        diff_w = np.linalg.norm(w-w_prev)
    return w, epoch


# Error de entropía cruzada
def errRL(x, y, w):
    return np.mean(np.log(1+np.exp(-y*np.matmul(x,w))))

# Error de clasificación aplicado a Regresión Logística
def errAccuracyRL(x, y, w):
    acc = 0
    for i in range(y.size):
        if clasificadorRL(w,x[i]) != y[i]:
            acc += 1
    return acc/y.size



lr = 0.01
error = 0.01
batch_size = 1
max_epochs = 10000

v_epochs = []
v_E_in = []
v_E_test = []
v_E_acc_in = []
v_E_acc_test = []

print('(la ejecución de los 100 experimentos me tarda entre 1 y 2 minutos)')
for rep in range(100):
    # Genero muestra uniforme con tamaño N=100 sobre el cuadrado [0,2]x[0,2]
    #   etiquetada a partir de una recta con ecuación y = a*x + b 
    a, b = simula_recta([0,2])
    x = simula_unif(100, 2, [0,2])
    x = np.concatenate((np.ones((x.shape[0],1),np.float64),x), axis=1)
    y = np.array([f(x[i][0],x[i][1],a,b) for i in range(x.shape[0])])
    
    # Aplico regresión logística
    w0 = np.zeros(x.shape[1])
    w, epochs = sgdRL(x, y, w0, lr, error, batch_size, max_epochs)
    v_epochs.append(epochs)
    
    # Calculo E_in y Acc_in
    v_E_in.append(errRL(x, y, w))
    v_E_acc_in.append(errAccuracyRL(x, y, w))
    
    # Genero muestra uniforme con tamaño N=1000 análoga a la anterior
    x = simula_unif(1000, 2, [0,2])
    x = np.concatenate((np.ones((x.shape[0],1),np.float64),x), axis=1)
    y = np.array([f(x[i][0],x[i][1],a,b) for i in range(x.shape[0])])
    
    # Calculo E_test y Acc_test
    v_E_test.append(errRL(x, y, w))
    v_E_acc_test.append(errAccuracyRL(x, y, w))


# Obtengo promedios
print('Valor medio de E_in: {}'.format(np.mean(np.asarray(v_E_in))))
print('Valor medio de E_test: {}'.format(np.mean(np.asarray(v_E_test))))
print('Valor medio de E_acc_in: {}'.format(np.mean(np.asarray(v_E_acc_in))))
print('Valor medio de E_acc_test: {}'.format(np.mean(np.asarray(v_E_acc_test))))
print('Valor medio de épocas necesarias para converger: {}'.format(np.mean(np.asarray(v_epochs))))


input("\n--- Pulsar tecla para continuar ---\n")








###############################################################################
###############################################################################
###############################################################################

#BONUS: Clasificación de Dígitos
print('BONUS')


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.scatter(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), c='blue', s=5, label='8')
ax.scatter(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), c='red', s=5, label='4')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

fig, ax = plt.subplots()
ax.scatter(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), c='blue', s=5, label='8')
ax.scatter(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), c='red', s=5, label='4')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")




# Función de error para Regresión Lineal (error cuadrático medio)
def errLinReg(x,y,w):
    return np.mean((np.matmul(x,w)-y)**2)

# Función de error para Pocket (fracción de elementos mal clasificados)
def errPocket(x, y, w):
    err = 0
    for i in range(y.size):
        if signo(np.dot(w,x[i])) != y[i]:
            err += 1
    return err/y.size



#LINEAR REGRESSION FOR CLASSIFICATION 

# Gradiente Descendente Estocástico para Regresión Lineal
def sgdLinReg(x, y, w0, lr, error, batch_size, max_epochs):
    w = np.copy(w0)
    w.astype(np.float64)
    epoch = 0
    while errLinReg(x,y,w) >= error and epoch < max_epochs:
        epoch += 1
        minibatches = np.split(np.random.permutation(np.arange(y.size)), 
                               range(batch_size,y.size,batch_size))
        for mb in minibatches:
            w_aux = np.copy(w)
            for k in range(w.size):
                w[k] -= lr*(2/mb.size)*np.dot(np.matmul(x[mb,:],w_aux)-y[mb], x[mb,k])
    return w

w0 = np.zeros(x.shape[1])
lr = 0.01
error = 0.001
batch_size = 50
max_epochs = 1000

w_linreg = sgdLinReg(x, y, w0, lr, error, batch_size, max_epochs)
E_in_reglin = errLinReg(x,y,w_linreg)
E_test_reglin = errLinReg(x_test,y_test,w_linreg)

print('E_in para Regresión Lineal:', E_in_reglin)
print('E_acc_in para Regresión Lineal:', errPocket(x,y,w_linreg))
dibujar_nube_etiq_con_recta(x[:,1:], y, -w_linreg[1]/w_linreg[2], -w_linreg[0]/w_linreg[2], 8, 4, 0, 1, -7, 0, 
                            'Solución de Regresión Lineal sobre la nube de puntos de entrenamiento')

input("\n--- Pulsar tecla para continuar ---\n")

print('E_test para Regresión Lineal:', E_test_reglin)
print('E_acc_test para Regresión Lineal:', errPocket(x_test,y_test,w_linreg))
dibujar_nube_etiq_con_recta(x_test[:,1:], y_test, -w_linreg[1]/w_linreg[2], -w_linreg[0]/w_linreg[2], 8, 4, 0, 1, -7, 0, 
                            'Solución de Regresión Lineal sobre la nube de puntos de test')

input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM

# Algoritmo Pocket
def pocket(x, y, w0, max_iter):
    w = np.copy(w0)
    w.astype(np.float64)
    w_best = np.copy(w)
    ein_best = errPocket(x,y,w_best)
    iterations = 0
    change = True
    while change and iterations < max_iter:
        change = False
        i = 0
        while i < y.size and iterations < max_iter:
            if signo(np.dot(w,x[i])) != y[i]:
                w += y[i]*x[i]
                change = True
                ein = errPocket(x,y,w)
                if ein < ein_best:
                    w_best = w
                    ein_best = ein
            i += 1
            iterations += 1
    return w_best

w0 = np.zeros(x.shape[1])
max_iter = 10000

w_pocket = pocket(x, y, w0, max_iter)
E_in_pocket = errPocket(x,y,w_pocket)
E_test_pocket = errPocket(x_test,y_test,w_pocket)

print('E_in para Pocket:', E_in_pocket)
dibujar_nube_etiq_con_recta(x[:,1:], y, -w_pocket[1]/w_pocket[2], -w_pocket[0]/w_pocket[2], 8, 4, 0, 1, -7, 0, 
                            'Solución de Pocket sobre la nube de puntos de entrenamiento')

input("\n--- Pulsar tecla para continuar ---\n")

print('E_test para Pocket:', E_test_pocket)
dibujar_nube_etiq_con_recta(x_test[:,1:], y_test, -w_pocket[1]/w_pocket[2], -w_pocket[0]/w_pocket[2], 8, 4, 0, 1, -7, 0, 
                            'Solución de Pocket sobre la nube de puntos de test')

input("\n--- Pulsar tecla para continuar ---\n")



# LINEAR REGRESSION + POCKET ALGORITHM

max_iter = 1000

w_lr_p = pocket(x, y, w_linreg, max_iter)
E_in_lr_p = errPocket(x,y,w_lr_p)
E_test_lr_p = errPocket(x_test,y_test,w_lr_p)

print('E_in para Reg. Lin. + Pocket:', E_in_lr_p)
dibujar_nube_etiq_con_recta(x[:,1:], y, -w_lr_p[1]/w_lr_p[2], -w_lr_p[0]/w_lr_p[2], 8, 4, 0, 1, -7, 0, 
                            'Solución de Reg. Lin. + Pocket sobre la nube de puntos de entrenamiento')

input("\n--- Pulsar tecla para continuar ---\n")

print('E_test para Reg. Lin. + Pocket:', E_test_lr_p)
dibujar_nube_etiq_con_recta(x_test[:,1:], y_test, -w_lr_p[1]/w_lr_p[2], -w_lr_p[0]/w_lr_p[2], 8, 4, 0, 1, -7, 0, 
                            'Solución de Reg. Lin. + Pocket sobre la nube de puntos de test')

input("\n--- Pulsar tecla para continuar ---\n")



#COTA SOBRE EL ERROR

d_vc = 3
delta = 0.05

epsilon_in = np.sqrt( (8/y.size)*np.log(4*((2*y.size)**d_vc+1)/delta) )
epsilon_test = np.sqrt( 1/(2*y_test.size)*np.log(2/delta) )

print('Cota de E_out para Regresión Lineal basada en E_in:\n\tE_out <=',E_in_reglin,'+',epsilon_in,'=',E_in_reglin+epsilon_in)
print('Cota de E_out para Regresión Lineal basada en E_test:\n\tE_out <=',E_test_reglin,'+',epsilon_test,'=',E_test_reglin+epsilon_test)
print()

print('Cota de E_out para Pocket basada en E_in:\n\tE_out <=',E_in_pocket,'+',epsilon_in,'=',E_in_pocket+epsilon_in)
print('Cota de E_out para Pocket basada en E_test:\n\tE_out <=',E_test_pocket,'+',epsilon_test,'=',E_test_pocket+epsilon_test)
print()

print('Cota de E_out para Reg. Lin. + Pocket basada en E_in:\n\tE_out <=',E_in_lr_p,'+',epsilon_in,'=',E_in_lr_p+epsilon_in)
print('Cota de E_out para Reg. Lin. + Pocket basada en E_test:\n\tE_out <=',E_test_lr_p,'+',epsilon_test,'=',E_test_lr_p+epsilon_test)
print()
