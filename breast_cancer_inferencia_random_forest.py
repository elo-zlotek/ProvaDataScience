import pandas as pd
from pickle import load

#Recebendo uma nova paciente desconhecida e determinando o indicativo de diagnóstico
nova_paciente = pd.DataFrame([{
    'age': '50-59',
    'menopause': 'ge40',
    'tumor-size': '20-24',
    'inv-nodes': '0-2',
    'node-caps': 'no',
    'deg-malig': 2,
    'breast': 'left',
    'breast-quad': 'left_up',
    'irradiat': 'no'
}])

#Codificando os atributos categóricos no mesmo padrão do treinamento
colunas_modelo = load(open('colunas_breast_cancer.model', 'rb'))
nova_paciente_encoded = pd.get_dummies(nova_paciente, dtype=float)
nova_paciente_encoded = nova_paciente_encoded.reindex(columns=colunas_modelo, fill_value=0.0)

#Carregando o normalizador salvo e normalizando a nova paciente
normalizador = load(open('normalizador_breast_cancer.model', 'rb'))
nova_paciente_normalizada = normalizador.transform(nova_paciente_encoded)
nova_paciente_normalizada = pd.DataFrame(nova_paciente_normalizada, columns=colunas_modelo)

#Carregando o modelo Random Forest
breast_cancer_rf = load(open('breast_cancer_rf.model', 'rb'))

#Classificando a nova paciente
diagnostico_nova_paciente = breast_cancer_rf.predict(nova_paciente_normalizada)
print('Diagnóstico (Random Forest):', diagnostico_nova_paciente[0])