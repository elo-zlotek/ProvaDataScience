import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from pickle import dump

#Carregando os dados
dados = pd.read_csv('breast-cancer.csv', sep=',', na_values='?')

#Preenchendo valores ausentes 
for coluna in dados.columns:
    if dados[coluna].isna().any():
        dados[coluna] = dados[coluna].fillna(dados[coluna].mode()[0])

#Separando atributos e classes
dados_atributos = dados.drop(columns=['Class'])
dados_classes = dados['Class']

#Convertendo atributos categóricos em numéricos 
dados_atributos_encoded = pd.get_dummies(dados_atributos, dtype=float)

#Salvando colunas geradas
dump(dados_atributos_encoded.columns.tolist(), open('colunas_breast_cancer.model', 'wb'))

#Balanceando os dados
resampler = SMOTE(random_state=42)
dados_atributos_balanceados, dados_classes_balanceados = resampler.fit_resample(
    dados_atributos_encoded, dados_classes
)

#Normalizando os dados
#MinMaxScaler
normalizador = MinMaxScaler()
#Treinando o modelo normalizador com os atributos balanceados
modelo_normalizador = normalizador.fit(dados_atributos_balanceados)
#Salvando o modelo normalizador para uso posterior
dump(modelo_normalizador, open('normalizador_breast_cancer.model', 'wb'))
#Normalizando os dados
dados_atributos_normalizados = modelo_normalizador.transform(dados_atributos_balanceados)
#Convertendo os dados normalizados para DataFrame novamente
dados_atributos_normalizados = pd.DataFrame(
    dados_atributos_normalizados, columns=dados_atributos_encoded.columns
)

#Hiperparametrização 
criterion = ['gini', 'entropy', 'log_loss']
splitter = ['best', 'random']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=5)]

decision_tree_grid = {
    'criterion': criterion,
    'splitter': splitter,
    'min_samples_split': min_samples_split,
    'random_state': [42]
}

tree = DecisionTreeClassifier()
tree_hiperparametros = RandomizedSearchCV(
    estimator=tree,
    param_distributions=decision_tree_grid,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)

#Rodando o random search
tree_hiperparametros.fit(dados_atributos_normalizados, dados_classes_balanceados)
print('Melhores hiperparâmetros (Decision Tree):', tree_hiperparametros.best_params_)

#Reconstruindo o classificador otimizado 
tree = DecisionTreeClassifier(**tree_hiperparametros.best_params_)
breast_cancer_tree = tree.fit(dados_atributos_normalizados, dados_classes_balanceados)

#Salvando para uso posterior
dump(breast_cancer_tree, open('breast_cancer_tree.model', 'wb'))

#Avaliando a acurácia do modelo
classes_preditas = breast_cancer_tree.predict(dados_atributos_normalizados)
acuracia_tree = accuracy_score(dados_classes_balanceados, classes_preditas)
print('Acuracia Decision Tree:', acuracia_tree)