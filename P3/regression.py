# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Semilla fijada
np.random.seed(1)



# Constantes
FILENAME = 'datos/data_regression.csv'

TEST_SIZE = 0.2

N_JOBS = 6

VISUALIZE_TRAIN_SET = False
CROSS_VALIDATION = False
CROSS_VALIDATION_KNR = False

VARIANCE_THRESHOLD = 1e-3
POL_DEGREE = 2
PCA_EXPLAINED_VARIANCE = 0.99999

K_SPLITS = 5
REG_PARAM_VALUES1 = [0.1, 1, 5, 10, 20]
REG_PARAM_VALUES2 = [4, 4.5, 5, 5.5, 6]
REG_PARAM = 5

NUM_NEIGHBORS_VALUES = [5, 10, 15, 20]
NUM_NEIGHBORS = 5



def readData(filename):
    X = []
    y = []
    with open(filename) as f:
        for line in f:
            attribs_label = line.split(",")
            X.append(attribs_label[:-1])
            y.append(attribs_label[-1])
    X.pop(0)
    y.pop(0)
    X = np.array(X, np.float64)
    y = np.array(y, np.float64)
    	
    return X, y



def tableCVResults(cv_results, precision=5):
    row = list(cv_results["params"][0].keys())+["mean E_in","mean E_cv"]
    format_row = "{:<20}" * len(row)
    print(format_row.format(*row))
    for i in range(len(cv_results["params"])):
        row = list(cv_results["params"][i].values())
        row.append(round(1-cv_results["mean_train_score"][i],precision))
        row.append(round(1-cv_results["mean_test_score"][i],precision))
        print(format_row.format(*row))



class PseudoinverseLinearRegression(BaseEstimator):
    def __init__(self, reg_param=0.0):
        self.reg_param = reg_param   # regularization parameter (lambda)
    
    # Ajuste del modelo
    def fit(self, X, y):
        inverse = np.linalg.inv(X.T @ X + self.reg_param*np.identity(X.shape[1]))
        self.w = np.dot( inverse, np.dot(X.T,y) )
    
    # Predicción de clases
    def predict(self, X):
        return np.dot(X,self.w)
    
    # Error Cuadrático Medio
    def mse(self, X, y):
        return metrics.mean_squared_error(y,self.predict(X))
    
    # Error Absoluto Medio
    def mae(self, X, y):
        return metrics.mean_absolute_error(y,self.predict(X))
    
    # Coeficiente de determinación (R^2)
    def R2(self, X, y):
        return 1-self.mse(X,y)/np.var(y)
    
    # Score: R^2
    def score(self, X, y):
        return self.R2(X,y)



class KNR(BaseEstimator):
    def __init__(self, num_neighbors=5, weight_function='uniform'):
        self.num_neighbors = num_neighbors   # número de vecinos
    
    # Ajuste del modelo
    def fit(self, X, y):
        self.model = KNeighborsRegressor(n_neighbors=self.num_neighbors,
                                         weights='uniform',
                                         n_jobs=N_JOBS)
        self.model.fit(X,y)
    
    # Predicción de clases
    def predict(self, X):
        return self.model.predict(X)
    
    # Error Cuadrático Medio
    def mse(self, X, y):
        return metrics.mean_squared_error(y,self.predict(X))
    
    # Error Absoluto Medio
    def mae(self, X, y):
        return metrics.mean_absolute_error(y,self.predict(X))
    
    # Coeficiente de determinación (R^2)
    def R2(self, X, y):
        return self.model.score(X,y)
    
    # Score: R^2
    def score(self, X, y):
        return self.R2(X,y)
    




if __name__ == "__main__":
    
    X, y = readData(FILENAME)
    
    # Separación train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    
    
    
    # Representación de datos en histogramas y en un espacio bidimensional mediante PCA y t-SNE
    if VISUALIZE_TRAIN_SET:
        print("#################################################################")
        print("########## VISUALIZACIÓN DEL CONJUNTO DE ENTRENAMIENTO ##########")
        print("#################################################################\n")
        
        print("Histograma con las temperaturas críticas y sus frec. absolutas")
        
        plt.hist(y_train, bins=37, density=False, cumulative=False)
        plt.xlabel("Temperatura crítica")
        plt.ylabel("Frecuencia absoluta")
        plt.title("Histograma con las temperaturas críticas y sus frec. absolutas")
        plt.grid(True)
        plt.show()
        
        input("\n--- Pulsar tecla para continuar ---\n")
        
        
        print("Histograma con las temperaturas críticas y sus frec. relativas acum.")
        
        plt.hist(y_train, bins=37, density=True, cumulative=True)
        plt.xlabel("Temperatura crítica")
        plt.ylabel("Frecuencia relativa acumulada")
        plt.title("Histograma con las temperaturas críticas y sus frec. relativas acum.")
        plt.grid(True)
        plt.show()
        
        input("\n--- Pulsar tecla para continuar ---\n")
        
        
        cmap='plasma'
        alpha=0.2
        
        X_train_95 = X_train[np.where(y_train<95.0)]
        y_train_95 = y_train[np.where(y_train<95.0)]
        
        print("Representación de los datos con reducción de dimensionalidad usando PCA")
        
        X_PCA = PCA(n_components=2, random_state=42).fit_transform(X_train_95)
        plt.scatter(X_PCA[:,0], X_PCA[:,1], c=y_train_95, cmap=cmap, alpha=alpha)
        plt.colorbar()
        plt.title("Representación de los datos en 2D usando PCA")
        plt.show()
        
        input("\n--- Pulsar tecla para continuar ---\n")
        
        
        print("Representación de los datos con reducción de dimensionalidad usando t-SNE")
        
        X_TSNE = TSNE(n_components=2, init=X_PCA).fit_transform(X_train_95)
        plt.scatter(X_TSNE[:,0], X_TSNE[:,1], c=y_train_95, cmap=cmap, alpha=alpha)
        plt.colorbar()
        plt.title("Representación de los datos en 2D usando t-SNE")
        plt.show()
        
        input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    print("##################################")
    print("########## PREPROCESADO ##########")
    print("##################################\n")
    
    
    # Matriz de coeficientes de correlación de Pearson con los datos iniciales
    # (previamente, eliminamos características constantes)
    correlation_matrix = np.corrcoef(np.transpose(VarianceThreshold().fit_transform(X_train)))
    print("Matriz de coeficientes de correlación de Pearson (datos iniciales)")
    plt.matshow(correlation_matrix, cmap='plasma')
    plt.colorbar()
    plt.title("Matriz de coef. de corr. de Pearson \n(datos iniciales)", pad=40.0)
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    
    print("Evolución del número de características:")
    print("\tDatos iniciales:", X_train.shape[1])
    
    # Eliminación de características con varianza muy baja
    variance_threshold = VarianceThreshold(VARIANCE_THRESHOLD)
    X_train = variance_threshold.fit_transform(X_train)
    X_test = variance_threshold.transform(X_test)
    print("\tVarianceThreshold:",X_train.shape[1])
    
    # Ampliación con características no lineales (polinomios con grado acotado)
    # También añade la característica asociada al término independiente
    polynomial_features = PolynomialFeatures(POL_DEGREE)
    X_train = polynomial_features.fit_transform(X_train)
    X_test = polynomial_features.transform(X_test)
    print("\tPolynomialFeatures:",X_train.shape[1])
    
    # Estándarización (características con media 0 y varianza 1)
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)
    print("\tStandardScaler:",X_train.shape[1])
    
    # Reducción de dimensionalidad mediante Análisis de Componentes Principales
    pca = PCA(n_components=PCA_EXPLAINED_VARIANCE)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print("\tPCA:",X_train.shape[1])
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    
    # Matriz de coeficientes de correlación de Pearson con los datos preprocesados
    correlation_matrix = np.corrcoef(np.transpose(X_train))
    print("Matriz de coeficientes de correlación de Pearson (datos preprocesados)")
    plt.matshow(correlation_matrix, cmap='plasma')
    plt.colorbar()
    plt.title("Matriz de coef. de corr. de Pearson \n(datos preprocesados)", pad=40.0)
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    # Creación del modelo de Regresión Lineal que usa la Pseudoinversa
    plr = PseudoinverseLinearRegression(reg_param=REG_PARAM)
    
    # Añado el término independiente a los elementos del conjunto de entrenamiento y test
    X_train = np.hstack(( np.ones((X_train.shape[0],1)), X_train ))
    X_test = np.hstack(( np.ones((X_test.shape[0],1)), X_test ))
    
    
    
    if CROSS_VALIDATION:
        print("######################################")
        print("########## CROSS-VALIDATION ##########")
        print("######################################\n")
        
        param_grid = {'reg_param':REG_PARAM_VALUES1}
        cv_searcher = GridSearchCV(plr, param_grid, n_jobs=N_JOBS, verbose=1, return_train_score=True)
        cv_searcher.fit(X_train, y_train)
        print()
        tableCVResults(cv_searcher.cv_results_)
        print()
        print("Mejores hiperparámetros:",cv_searcher.best_params_)
        print("E_in medio:",round(1-cv_searcher.cv_results_["mean_train_score"][np.where(cv_searcher.cv_results_["rank_test_score"]==1)[0][0]],5))
        print("E_cv medio:",round(1-cv_searcher.best_score_,5))
        print()
        
        param_grid = {'reg_param':REG_PARAM_VALUES2}
        cv_searcher = GridSearchCV(plr, param_grid, n_jobs=N_JOBS, verbose=1, return_train_score=True)
        cv_searcher.fit(X_train, y_train)
        print()
        tableCVResults(cv_searcher.cv_results_)
        print()
        print("Mejores hiperparámetros:",cv_searcher.best_params_)
        print("E_in medio:",round(1-cv_searcher.cv_results_["mean_train_score"][np.where(cv_searcher.cv_results_["rank_test_score"]==1)[0][0]],5))
        print("E_cv medio:",round(1-cv_searcher.best_score_,5))
        print()
        
        plr.set_params(**(cv_searcher.best_params_))
        
        input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    print("##########################################################")
    print("########## EVALUACIÓN DE LA HIPÓTESIS FINAL ##############")
    print("##########################################################\n")
    
    plr.fit(X_train, y_train)
    
    print("\nE_in =",round(1-plr.R2(X_train,y_train),5))
    print("R²_in =",round(plr.R2(X_train,y_train),5))
    print("MAE_in =",round(plr.mae(X_train,y_train),5))
    
    print("\nE_test =",round(1-plr.R2(X_test,y_test),5))
    print("R²_test =",round(plr.R2(X_test,y_test),5))
    print("MAE_test:",round(plr.mae(X_test,y_test),5))
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    # Creación del modelo KNR
    knr = KNR(num_neighbors=NUM_NEIGHBORS)
    
    # Elimino el término independiente de los elementos del conjunto de entrenamiento y test
    X_train = X_train[:,1:]
    X_test = X_test[:,1:]
    
    
    
    if CROSS_VALIDATION_KNR:
        print("############################################")
        print("########## CROSS-VALIDATION (KNR) ##########")
        print("############################################\n")
        
        param_grid = {'num_neighbors':NUM_NEIGHBORS_VALUES}
        cv_searcher = GridSearchCV(knr, param_grid, n_jobs=N_JOBS, verbose=1, return_train_score=True)
        cv_searcher.fit(X_train, y_train)
        print()
        tableCVResults(cv_searcher.cv_results_)
        print()
        print("Mejores hiperparámetros:",cv_searcher.best_params_)
        print("E_in medio:",round(1-cv_searcher.cv_results_["mean_train_score"][np.where(cv_searcher.cv_results_["rank_test_score"]==1)[0][0]],5))
        print("E_cv medio:",round(1-cv_searcher.best_score_,5))
        print()
        
        knr.set_params(**(cv_searcher.best_params_))
        
        input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    print("################################################################")
    print("########## EVALUACIÓN DE LA HIPÓTESIS FINAL (KNR) ##############")
    print("################################################################\n")
    
    knr.fit(X_train,y_train)
    
    print("\nE_in =",round(1-knr.R2(X_train,y_train),5))
    print("R²_in =",round(knr.R2(X_train,y_train),5))
    print("MAE_in =",round(knr.mae(X_train,y_train),5))
    
    print("\nE_test =",round(1-knr.R2(X_test,y_test),5))
    print("R²_test =",round(knr.R2(X_test,y_test),5))
    print("MAE_test:",round(knr.mae(X_test,y_test),5))