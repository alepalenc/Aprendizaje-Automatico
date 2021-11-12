# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Fijamos la semilla
np.random.seed(1)



# Constantes
FILENAME = 'datos/data_classification.txt'

TEST_SIZE = 0.2

N_JOBS = 6

VISUALIZE_TRAIN_SET = False
CROSS_VALIDATION = False
CROSS_VALIDATION_SVC = False

VARIANCE_THRESHOLD = 1e-7
POL_DEGREE = 2
PCA_EXPLAINED_VARIANCE = 0.999

K_SPLITS = 5
REG_PARAM_VALUES1 = [0.01, 0.1, 1, 10]
REG_PARAM_VALUES2 = [0.01, 0.1, 1, 10]
REG_PARAM_VALUES3 = [0.05, 0.1, 0.3, 0.5, 0.7]
PENALTY_VALUES1 = ['l1','l2']
PENALTY_VALUES2 = ['l2']
PENALTY_VALUES3 = ['l2']
SOLVER1 = 'saga'
SOLVER2 = 'lbfgs'
SOLVER3 = 'lbfgs'

REG_PARAM = 0.1
PENALTY = 'l2'
SOLVER = 'lbfgs'

REG_PARAM_SVC_VALUES = [0.01, 0.1, 1, 10]
GAMMA_SVC_VALUES = [0.001, 'scale']
REG_PARAM_SVC = 0.01
GAMMA_SVC = 0.001






def readData(filename):
    X = []
    y = []
    with open(filename) as f:
        for line in f:
            chars_label = [float(s) for s in line.split(" ")]
            X.append(chars_label[:-1])
            y.append(chars_label[-1])
    
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



def sensitivity(confusion_matrix):
    sensit = []
    for k in range(confusion_matrix.shape[0]):
        sensit.append(round(confusion_matrix[k][k]/np.sum(confusion_matrix[:,k]),5))
    return np.array(sensit)


def specificity(confusion_matrix):
    specif = []
    suma_diag = np.sum([confusion_matrix[k][k] for k in range(confusion_matrix.shape[0])])
    for k in range(confusion_matrix.shape[0]):
        specif.append(round((suma_diag-confusion_matrix[k][k])/(1-np.sum(confusion_matrix[:,k])),5))
    return np.array(specif)
    



class MultinomialLogisticRegression(BaseEstimator):
    def __init__(self, reg_param=1.0, penalty='l2'):
        self.reg_param = reg_param   # parámetro de regularización (lambda)
        if self.reg_param <= 0.0:
            self.reg_param = 1.0
        self.penalty = penalty   # norma usada en regularización
    
    # Ajuste del modelo
    def fit(self, X, y, solver='lbfgs', max_iter=1000):
        self.model = LogisticRegression(penalty=self.penalty,
                                        tol=0.05,
                                        C=1/self.reg_param,
                                        solver=solver,
                                        max_iter=max_iter,
                                        multi_class='multinomial',
                                        n_jobs=N_JOBS)
        self.model.fit(X,y)
    
    # Predicción de clases
    def predict(self, X):
        return self.model.predict(X)
    
    # Predicción de la probabilidad de cada clase mediante softmax
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    # Función de pérdida (error de entropía cruzada)
    def log_loss(self, X, y):
        return metrics.log_loss(y,self.predict_proba(X))
    
    # Accuracy
    def accuracy(self, X, y):
        return self.model.score(X,y)
    
    # Score: Accuracy
    def score(self, X, y):
        return self.accuracy(X,y)



class SupportVectorClassification(BaseEstimator):
    def __init__(self, reg_param=1.0, gamma='scale'):
        self.reg_param = reg_param   # parámetro de regularización (lambda)
        self.gamma = gamma   # coeficiente que multiplica al kernel
    
    # Ajuste del modelo
    def fit(self, X, y):
        self.model = SVC(C=1/self.reg_param,
                         kernel='rbf',
                         gamma = self.gamma,
                         tol=1e-3,
                         decision_function_shape='ovr')
        self.model.fit(X,y)
    
    # Predicción de clases
    def predict(self, X):
        return self.model.predict(X)
    
    # Accuracy
    def accuracy(self, X, y):
        return self.model.score(X,y)
    
    # Score: Accuracy
    def score(self, X, y):
        return self.accuracy(X,y)



if __name__ == "__main__":

    X, y = readData(FILENAME)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    
    # Representación de datos en 2 dimensiones
    if VISUALIZE_TRAIN_SET:
        print("#################################################################")
        print("########## VISUALIZACIÓN DEL CONJUNTO DE ENTRENAMIENTO ##########")
        print("#################################################################\n")
        
        print("Histograma que agrupa las característica según su varianza")
        
        chars_var = np.array([np.var(X_train.T[i]) for i in range(X_train.shape[1])])
        plt.hist(chars_var, bins=[0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
        plt.xlabel("Varianza")
        plt.ylabel("Número de características")
        plt.title("Histograma que agrupa las característica según su varianza")
        plt.grid(True)
        plt.xscale('log')
        plt.show()
        
        input("\n--- Pulsar tecla para continuar ---\n")
        
        
        print("Diagrama de barras que agrupa a los elementos por clases")
        
        class_freq = np.array([np.count_nonzero(y_train==k) for k in np.sort(np.unique(y_train))])
        classes = [str(int(k)) for k in np.sort(np.unique(y_train))]
        plt.bar(x=classes, height=class_freq, width=0.5)
        plt.xlabel("Clases")
        plt.ylabel("Frecuencia absoluta")
        plt.title("Diagrama de barras que agrupa a los elementos por clases")
        plt.grid(True)
        plt.show()
        
        input("\n--- Pulsar tecla para continuar ---\n")
        
        
        cmap='Paired'
        alpha=0.4
        
        print("Representación de los datos con reducción de dimensionalidad usando PCA")
        
        X_PCA = PCA(n_components=2, random_state=32).fit_transform(X)
        plt.scatter(X_PCA[:,0], X_PCA[:,1], c=y, cmap=cmap, alpha=alpha)
        cmap=cm.get_cmap('Paired')
        proxys=[]
        labels=[]
        for label in np.sort((np.unique(y_train))).astype(np.int8):
            proxys.append(Line2D([0],[0], linestyle='none', c=cmap(label/10), marker='o'))
            labels.append(str(label))
        plt.xlim((-3,30))
        plt.ylim((-75,30))
        plt.legend(proxys, labels, numpoints = 1, framealpha=0.5)
        plt.title("Representación de los datos en 2D usando PCA")
        plt.show()
        
        input("\n--- Pulsar tecla para continuar ---\n")
        
        
        print("Representación de los datos con reducción de dimensionalidad usando t-SNE")
        
        X_TSNE = TSNE(n_components=2, init=X_PCA).fit_transform(X)
        plt.scatter(X_TSNE[:,0], X_TSNE[:,1], c=y, cmap=cmap, alpha=alpha)
        proxys=[]
        labels=[]
        for label in np.sort((np.unique(y_train))).astype(np.int8):
            proxys.append(Line2D([0],[0], linestyle='none', c=cmap(label/10), marker='o'))
            labels.append(str(label))
        plt.xlim((-60,70))
        plt.ylim((-70,65))
        plt.legend(proxys, labels, numpoints = 1, framealpha=0.5)
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
    
    
        
    
    # Creación del modelo de Regresión Logística Multinomial
    mlr = MultinomialLogisticRegression(reg_param=REG_PARAM, 
                                                        penalty=PENALTY)
    
    # Añado el término independiente a los elementos del conjunto de entrenamiento y test
    X_train = np.hstack(( np.ones((X_train.shape[0],1)), X_train ))
    X_test = np.hstack(( np.ones((X_test.shape[0],1)), X_test ))
    
    
    
    if CROSS_VALIDATION:
        print("######################################")
        print("########## CROSS-VALIDATION ##########")
        print("######################################\n")
        
        param_grid = {'reg_param':REG_PARAM_VALUES1,
                      'penalty':PENALTY_VALUES1}
        cv_searcher = GridSearchCV(mlr, param_grid, n_jobs=N_JOBS, verbose=1, return_train_score=True)
        cv_searcher.fit(X_train, y_train, solver=SOLVER1)
        print()
        tableCVResults(cv_searcher.cv_results_)
        print()
        print("Algoritmo usado:",SOLVER1)
        print("Mejores hiperparámetros:",cv_searcher.best_params_)
        print("E_in medio:",round(1-cv_searcher.cv_results_["mean_train_score"][np.where(cv_searcher.cv_results_["rank_test_score"]==1)[0][0]],5))
        print("E_cv medio:",round(1-cv_searcher.best_score_,5))
        print()
        
        param_grid = {'reg_param':REG_PARAM_VALUES2,
                      'penalty':PENALTY_VALUES2}
        cv_searcher = GridSearchCV(mlr, param_grid, n_jobs=N_JOBS, verbose=1, return_train_score=True)
        cv_searcher.fit(X_train, y_train, solver=SOLVER2)
        print()
        tableCVResults(cv_searcher.cv_results_)
        print()
        print("Algoritmo usado:",SOLVER2)
        print("Mejores hiperparámetros:",cv_searcher.best_params_)
        print("E_in medio:",round(1-cv_searcher.cv_results_["mean_train_score"][np.where(cv_searcher.cv_results_["rank_test_score"]==1)[0][0]],5))
        print("E_cv medio:",round(1-cv_searcher.best_score_,5))
        print()
        
        param_grid = {'reg_param':REG_PARAM_VALUES3,
                      'penalty':PENALTY_VALUES3}
        cv_searcher = GridSearchCV(mlr, param_grid, n_jobs=N_JOBS, verbose=1, return_train_score=True)
        cv_searcher.fit(X_train, y_train, solver=SOLVER3)
        print()
        tableCVResults(cv_searcher.cv_results_)
        print()
        print("Algoritmo usado:",SOLVER3)
        print("Mejores hiperparámetros:",cv_searcher.best_params_)
        print("E_in medio:",round(1-cv_searcher.cv_results_["mean_train_score"][np.where(cv_searcher.cv_results_["rank_test_score"]==1)[0][0]],5))
        print("E_cv medio:",round(1-cv_searcher.best_score_,5))
        print()
        
        mlr.set_params(**(cv_searcher.best_params_))
        
        input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    print("##########################################################")
    print("########## EVALUACIÓN DE LA HIPÓTESIS FINAL ##############")
    print("##########################################################\n")
    
    mlr.fit(X_train, y_train, solver=SOLVER)
    
    print("\nE_in =",round(1-mlr.accuracy(X_train, y_train),5))
    print("Accuracy_in =",round(mlr.accuracy(X_train, y_train),5))
    confusion_matrix = metrics.confusion_matrix(y_train,
                                                mlr.predict(X_train),
                                                normalize='all')
    print("Sensitivity_in =",sensitivity(confusion_matrix))
    print("Specificity_in =",specificity(confusion_matrix))
    print("Matriz de confusión (train)")
    plt.matshow(confusion_matrix, cmap='plasma')
    plt.colorbar()
    plt.title("Matriz de confusión (train)", pad=20.0)
    plt.show()
    
    print("\nE_test =",round(1-mlr.accuracy(X_test, y_test),5))
    print("Accuracy_test =",round(mlr.accuracy(X_test, y_test),5))
    confusion_matrix = metrics.confusion_matrix(y_test,
                                                mlr.predict(X_test),
                                                normalize='all')
    print("Sensitivity_test =",sensitivity(confusion_matrix))
    print("Specificity_test =",specificity(confusion_matrix))
    print("Matriz de confusión (test)")
    plt.matshow(confusion_matrix, cmap='plasma')
    plt.colorbar()
    plt.title("Matriz de confusión (test)", pad=20.0)
    plt.show()
    
    input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    # Creación del modelo SVC
    svc = SupportVectorClassification(reg_param=REG_PARAM_SVC, gamma=GAMMA_SVC)
    
    
    
    if CROSS_VALIDATION_SVC:
        print("############################################")
        print("########## CROSS-VALIDATION (SVC) ##########")
        print("############################################\n")
        
        param_grid = {'reg_param':REG_PARAM_SVC_VALUES,
                      'gamma':GAMMA_SVC_VALUES}
        cv_searcher = GridSearchCV(svc, param_grid, n_jobs=N_JOBS, verbose=100, return_train_score=True)
        cv_searcher.fit(X_train, y_train)
        print()
        tableCVResults(cv_searcher.cv_results_)
        print()
        print("Mejores hiperparámetros:",cv_searcher.best_params_)
        print("E_in medio:",round(1-cv_searcher.cv_results_["mean_train_score"][np.where(cv_searcher.cv_results_["rank_test_score"]==1)[0][0]],5))
        print("E_cv medio:",round(1-cv_searcher.best_score_,5))
        print()
        
        svc.set_params(**(cv_searcher.best_params_))
        
        input("\n--- Pulsar tecla para continuar ---\n")
    
    
    
    print("################################################################")
    print("########## EVALUACIÓN DE LA HIPÓTESIS FINAL (SVC) ##############")
    print("################################################################\n")
    
    svc.fit(X_train, y_train)
    
    print("\nE_in =",round(1-svc.accuracy(X_train, y_train),5))
    print("Accuracy_in =",round(svc.accuracy(X_train, y_train),5))
    confusion_matrix = metrics.confusion_matrix(y_train,
                                                svc.predict(X_train),
                                                normalize='all')
    print("Sensitivity_in =",sensitivity(confusion_matrix))
    print("Specificity_in =",specificity(confusion_matrix))
    print("Matriz de confusión (train)")
    plt.matshow(confusion_matrix, cmap='plasma')
    plt.colorbar()
    plt.title("Matriz de confusión (train)", pad=20.0)
    plt.show()
    
    print("\nE_test =",round(1-svc.accuracy(X_test, y_test),5))
    print("Accuracy_test =",round(svc.accuracy(X_test, y_test),5))
    confusion_matrix = metrics.confusion_matrix(y_test,
                                                svc.predict(X_test),
                                                normalize='all')
    print("Sensitivity_test =",sensitivity(confusion_matrix))
    print("Specificity_test =",specificity(confusion_matrix))
    print("Matriz de confusión (test)")
    plt.matshow(confusion_matrix, cmap='plasma')
    plt.colorbar()
    plt.title("Matriz de confusión (test)", pad=20.0)
    plt.show()
    
    
    
    
    