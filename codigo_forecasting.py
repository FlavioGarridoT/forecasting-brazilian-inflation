
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

from dieboldmariano import dm_test

import warnings 
warnings.filterwarnings('ignore')
#####


dados = pd.read_excel("dados/DADOS.xlsx")
ipca_2018e2019 = pd.read_excel("dados/IPCA_2018e2019.xlsx") ## Esses são os dados fora da amostra

# Pegando os horizontes temporais dos dados fora da amostra para fazer os erros de previsão
ipca_OOB_6meses = ipca_2018e2019[0:6]['IPCA']
ipca_OOB_9meses = ipca_2018e2019[0:9]['IPCA']
ipca_OOB_12meses = ipca_2018e2019[0:12]['IPCA']
ipca_OOB_24meses = ipca_2018e2019[0:24]['IPCA']


# Teste Dickey-Fuller Aumentado -- apenas testando para o IPCA, não irei refazer pois já havia feito no R
# teste_adf_ipca = adfuller(dados['ipca'].values)
# teste_adf_ipca[0] # estatística do teste
# teste_adf_ipca[1] # p-valor

# ESSAS TRANSFORMAÇÕES SÃO PARA DEIXAR OS DADOS COMPARÁVEIS AO ARTIGO DE ARAÚJO & GAGLIANONE (2023)
# Primeira diferenca -> selic, tjlp, compulsorio, indice confianca, res_primario, res_primario_pib
# Segunda diferenca -> DLSP, DLSP_PIB
# ln(x) -> 
# delta(ln(x)) -> base monetaria, embi, prodVeiculos, IBC-BR, Importacoes, exportacoes, crb, petroleo brent, M1, ibovespa, cambio
# delta²(ln(x)) -> PIB, reservas internacionais, M2, M3, M4


# SÉRIES QUE SÓ PRECISAM FAZER A PRIMEIRA DIFERENÇA  # 143 linhas
selic_estacionaria = dados['selic meta'].diff().dropna() #print(selic_estacionaria.values)
tjlp_estacionaria = dados['tjlp'].diff().dropna()
compulsorio_estacionaria = dados['compulsorio'].diff().dropna()
indice_confianca_estacionaria = dados['indice confianca consumidor'].diff().dropna()
resultado_primario_estacionaria = dados['ResPrimario'].diff().dropna()
resultado_primario_pib_estacionaria = dados['ResPrimarioPIB'].diff().dropna()

# SEGUNDA DIFERENÇA 142 linhas
DLSP_estacionaria = dados['DLSP'].diff(2).dropna()
DLSP_pib_estacionaria = dados['DLSPPIB'].diff(2).dropna()

# DELTA(LN(X))                    143 linhas 
base_monetaria_estacionaria = np.log(dados['base monetaria']).diff().dropna()
embi_estacionaria = np.log(dados['embi+']).diff().dropna()
prodVeiculos_estacionaria = np.log(dados['prod veiculos']).diff().dropna()
IBCbr_estacionaria = np.log(dados['ibc-br']).diff().dropna()
importacoes_estacionaria = np.log(dados['importacoes']).diff().dropna()
exportacoes_estacionaria = np.log(dados['exportacoes']).diff().dropna()
crb_estacionaria = np.log(dados['crb']).diff().dropna()
petroleo_estacionaria = np.log(dados['petroleo brent']).diff().dropna()
m1_estacionaria = np.log(dados['M1']).diff().dropna()
ibovespa_estacionaria = np.log(dados['ibovespa']).diff().dropna()
cambio_estacionaria = np.log(dados['cambio']).diff().dropna()

# DELTA²(LN(X))
pib_estacionaria = np.log(dados['pib']).diff(2).dropna()
reservas_estacionaria = np.log(dados['reservas internacionais']).diff(2).dropna()     # 142 linhas
m2_estacionaria = np.log(dados['M2']).diff(2).dropna()
m3_estacionaria = np.log(dados['M3']).diff(2).dropna()
m4_estacionaria = np.log(dados['M4']).diff(2).dropna()

# Pegando o IPCA com 3 defasagens por conta da autocorrelação  144 linhas 
inflacao_3lag = dados['ipca'].diff(3)
#plt.plot(inflacao_3lag)
#plt.show()

# Juntando todas as variáveis novamente
# Criei dois grupos: um das variáveis que já eram estacionárias, e outro das que foram transformadas
df1 = dados.iloc[:, 0:17].join(dados['variacao mensal varejo - pmc'])
df2 = pd.concat([selic_estacionaria,tjlp_estacionaria,compulsorio_estacionaria,indice_confianca_estacionaria,resultado_primario_estacionaria,
           resultado_primario_pib_estacionaria,DLSP_estacionaria,DLSP_pib_estacionaria,base_monetaria_estacionaria,embi_estacionaria,
           prodVeiculos_estacionaria,IBCbr_estacionaria,importacoes_estacionaria,exportacoes_estacionaria,crb_estacionaria,
           petroleo_estacionaria,m1_estacionaria,ibovespa_estacionaria,cambio_estacionaria,pib_estacionaria,reservas_estacionaria,
           m2_estacionaria,m3_estacionaria,m4_estacionaria,inflacao_3lag] , axis = 1)
#dados['M1']
#dados_transformados['M1']

# Juntando então dataframe 1 e o dataframe 2
dados_transformados = pd.concat([df1, df2], axis = 1)
dados_transformados = dados_transformados.iloc[:, 1:-1]
dados_transformados

dados['data']

# Dividindo a base entre treino e teste já agora

X = dados_transformados.drop('ipca', axis = 1)
y = dados_transformados['ipca']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 555)

# Para fazer o eixo X dos gráficos
eixoX_graph_treino = dados.loc[:99, 'data'] ## eixo X do gráfico. Pegando as 100 primeiras linhas da coluna data
eixoX_graph_treino

eixoX_graph_teste = dados.loc[100:, 'data']
eixoX_graph_teste



# Criando um pipeline para agilizar e preencher os valores vazios e normalizar os dados
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())    
])


X_treino_transformed = num_pipeline.fit_transform(X_train)
X_teste_transformed = num_pipeline.transform(X_test)



################################################################ CRIAÇÃO DE MODELOS ######################################

########################################### MODELOS DE SÉRIES TEMPORAIS


# ARIMA
AR1 = ARIMA(y_train, order=(1,0,0), seasonal_order=(1,0,0,12)).fit()
AR2 = ARIMA(y_train, order=(2,0,0), seasonal_order=(2,0,0,12)).fit()
AR3 = ARIMA(y_train, order=(3,0,0), seasonal_order=(1,1,1,12)).fit()
BC_ARMA = ARIMA(y_train, order=(4,0,3), seasonal_order=(2,2,2,12)).fit() ## ARMA(4,3) usado pelo Banco Central

# Fazendo previsões
# AR1
previsao_6meses_AR1 = AR1.forecast(steps = 6) #6 meses
previsao_9meses_AR1 = AR1.forecast(steps = 9) 
previsao_12meses_AR1 = AR1.forecast(steps = 12)
previsao_24meses_AR1 = AR1.forecast(steps = 24)

# AR2 
previsao_6meses_AR2 = AR2.forecast(steps = 6) #6 meses
previsao_9meses_AR2 = AR2.forecast(steps = 9) 
previsao_12meses_AR2 = AR2.forecast(steps = 12)
previsao_24meses_AR2 = AR2.forecast(steps = 24)

# AR3
previsao_6meses_AR3 = AR3.forecast(steps = 6) #6 meses
previsao_9meses_AR3 = AR3.forecast(steps = 9) 
previsao_12meses_AR3 = AR3.forecast(steps = 12)
previsao_24meses_AR3 = AR3.forecast(steps = 24)

# ARMA(4,3)
previsao_6meses_ARMA = BC_ARMA.forecast(steps = 6) #6 meses
previsao_9meses_ARMA = BC_ARMA.forecast(steps = 9) 
previsao_12meses_ARMA = BC_ARMA.forecast(steps = 12)
previsao_24meses_ARMA = BC_ARMA.forecast(steps = 24)


# plt.figure(figsize=(14,7))
# plt.plot(eixoX_graph_treino,y_train, label = 'IPCA treino', color = 'red')
# plt.plot(eixoX_graph_teste,y_test, label = 'IPCA teste', color = 'green')
# plt.plot(eixoX_graph_teste, previsao_6meses_AR1, label = 'Prev AR1 - 6 meses', color = 'blue')
# plt.title('Previsão AR1 - 6 meses')
# plt.xlabel('Date')
# plt.ylabel('Tx Inflação')
# plt.legend()
# plt.show()


# Pegando o erro dos modelos (MSE e RMSE). Só fiz AR1 e ARMA pq é o que eu usei na dissertação.
# AR1 
# 6 meses
mean_squared_error(ipca_OOB_6meses, previsao_6meses_AR1)
np.sqrt(mean_squared_error(ipca_OOB_6meses, previsao_6meses_AR1))
# 9 meses
mean_squared_error(ipca_OOB_9meses, previsao_9meses_AR1)
np.sqrt(mean_squared_error(ipca_OOB_9meses, previsao_9meses_AR1))
# 12 meses
mean_squared_error(ipca_OOB_12meses, previsao_12meses_AR1)
np.sqrt(mean_squared_error(ipca_OOB_12meses, previsao_12meses_AR1))
# 24 meses 
mean_squared_error(ipca_OOB_24meses, previsao_24meses_AR1)
np.sqrt(mean_squared_error(ipca_OOB_24meses, previsao_24meses_AR1))


# ARMA (4,3)
# 6 meses
mean_squared_error(ipca_OOB_6meses, previsao_6meses_ARMA)
np.sqrt(mean_squared_error(ipca_OOB_6meses, previsao_6meses_ARMA))
# 9 meses
mean_squared_error(ipca_OOB_9meses, previsao_9meses_ARMA)
np.sqrt(mean_squared_error(ipca_OOB_9meses, previsao_9meses_ARMA))
# 12 meses
mean_squared_error(ipca_OOB_12meses, previsao_12meses_ARMA)
np.sqrt(mean_squared_error(ipca_OOB_12meses, previsao_12meses_ARMA))
# 24 meses
mean_squared_error(ipca_OOB_24meses, previsao_24meses_ARMA)
np.sqrt(mean_squared_error(ipca_OOB_24meses, previsao_24meses_ARMA))



############################################################ MACHINE LEARNING

## Como já tinha feito a tunagem de hiperparâmetros no R, não tem pq fazer aqui
## Só irei criar os modelos com os parâmetros iguais aos que a validação cruzada achou no R


########################################## KNN
# criando  e dando fit no modelo
modelo_knn = KNeighborsRegressor(n_neighbors = 5)  
modelo_knn.fit(X_treino_transformed, y_train)
# fazendo previsão
previsao_knn = modelo_knn.predict(X_teste_transformed)
previsao_knn

knn_6meses = previsao_knn[0:6]
knn_9meses = previsao_knn[0:9]
knn_12meses = previsao_knn[0:12]
knn_24meses = previsao_knn[0:24]

# erros de previsao
# 6 meses
mean_squared_error(ipca_OOB_6meses, knn_6meses)
np.sqrt(mean_squared_error(ipca_OOB_6meses, knn_6meses))
# 9 meses
mean_squared_error(ipca_OOB_9meses, knn_9meses)
np.sqrt(mean_squared_error(ipca_OOB_9meses, knn_9meses))
# 12 meses
mean_squared_error(ipca_OOB_12meses, knn_12meses)
np.sqrt(mean_squared_error(ipca_OOB_12meses, knn_12meses))
# 24 meses
mean_squared_error(ipca_OOB_24meses, knn_24meses)
np.sqrt(mean_squared_error(ipca_OOB_24meses, knn_24meses))



############################################## SVM
modelo_svm = SVR(kernel='linear', C = 0.1, gamma = 0.5, epsilon = 0.2)
modelo_svm.fit(X_treino_transformed, y_train)
# fazendo previsão
previsao_svm = modelo_svm.predict(X_teste_transformed)
previsao_svm

svm_6meses = previsao_svm[0:6]
svm_9meses = previsao_svm[0:9]
svm_12meses = previsao_svm[0:12]
svm_24meses = previsao_svm[0:24]

# erros de previsao
# 6 meses
mean_squared_error(ipca_OOB_6meses, svm_6meses)
np.sqrt(mean_squared_error(ipca_OOB_6meses, svm_6meses))
# 9 meses
mean_squared_error(ipca_OOB_9meses, svm_9meses)
np.sqrt(mean_squared_error(ipca_OOB_9meses, svm_9meses))
# 12 meses
mean_squared_error(ipca_OOB_12meses, svm_12meses)
np.sqrt(mean_squared_error(ipca_OOB_12meses, svm_12meses))
# 24 meses
mean_squared_error(ipca_OOB_24meses, svm_24meses)
np.sqrt(mean_squared_error(ipca_OOB_24meses, svm_24meses))



############################################### LIGHT GBM
modelo_lgbm = LGBMRegressor(n_estimators=200, min_child_samples=3, max_depth=2, learning_rate = 0.1)
modelo_lgbm.fit(X_treino_transformed, y_train)
# fazendo previsão
previsao_lgbm = modelo_lgbm.predict(X_teste_transformed)
previsao_lgbm

# FAZENDO CROSS-VALIDATION PARA ACHAR OS MELHORES PARAMETROS DO LGBM -- teste apenas

# lgbm_reg = LGBMRegressor(random_state=555)

# param_grid_lgbm = {
#     'max_depth' : [2,3,5] ,
#     'num_leaves' : [20, 30],
#     'learning_rate' : [0.05, 0.1],
#     'n_estimators' : [100, 150, 200]
# }

#lgbm_cv = GridSearchCV(lgbm_reg, param_grid_lgbm, cv = 3, scoring = 'neg_root_mean_squared_error', n_jobs = -1)
#lgbm_cv.fit(X_treino_transformed, y_train)
#lgbm_cv.best_params_
# ###  {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 100, 'num_leaves': 20}

#lgbm_cv.predict(X_teste_transformed)

lgbm_6meses = previsao_lgbm[0:6]
lgbm_9meses = previsao_lgbm[0:9]
lgbm_12meses = previsao_lgbm[0:12]
lgbm_24meses = previsao_lgbm[0:24]


# erros de previsao
# 6 meses
mean_squared_error(ipca_OOB_6meses, lgbm_6meses)
np.sqrt(mean_squared_error(ipca_OOB_6meses, lgbm_6meses))
# 9 meses
mean_squared_error(ipca_OOB_9meses, lgbm_9meses)
np.sqrt(mean_squared_error(ipca_OOB_9meses, lgbm_9meses))
# 12 meses
mean_squared_error(ipca_OOB_12meses, lgbm_12meses)
np.sqrt(mean_squared_error(ipca_OOB_12meses, lgbm_12meses))
# 24 meses
mean_squared_error(ipca_OOB_24meses, lgbm_24meses)
np.sqrt(mean_squared_error(ipca_OOB_24meses, lgbm_24meses))




############################################### TESTE DE DIEBOLD MARIANO ######################################
### O output do método retorna uma tupla com dois valores
### 1) a estatística de teste (DM) e 2) p-valor

# AR1 x ARMA 
# 6 meses
dm_test(ipca_OOB_6meses, previsao_6meses_AR1, previsao_6meses_ARMA)
# 9 meses
dm_test(ipca_OOB_9meses, previsao_9meses_AR1, previsao_9meses_ARMA)
# 12 meses
dm_test(ipca_OOB_12meses, previsao_12meses_AR1, previsao_12meses_ARMA)
# 24 meses
dm_test(ipca_OOB_24meses, previsao_24meses_AR1, previsao_24meses_ARMA)

# AR1 x KNN
dm_test(ipca_OOB_6meses, previsao_6meses_AR1, knn_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, previsao_9meses_AR1, knn_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, previsao_12meses_AR1, knn_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, previsao_24meses_AR1, knn_24meses)

# AR1 x SVM
dm_test(ipca_OOB_6meses, previsao_6meses_AR1, svm_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, previsao_9meses_AR1, svm_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, previsao_12meses_AR1, svm_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, previsao_24meses_AR1, svm_24meses)

# AR1 x LGBM 
dm_test(ipca_OOB_6meses, previsao_6meses_AR1, lgbm_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, previsao_9meses_AR1, lgbm_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, previsao_12meses_AR1, lgbm_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, previsao_24meses_AR1, lgbm_24meses)



# ARMA x KNN
dm_test(ipca_OOB_6meses, previsao_6meses_ARMA, knn_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, previsao_9meses_ARMA, knn_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, previsao_12meses_ARMA, knn_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, previsao_24meses_ARMA, knn_24meses)

# ARMA x SVM 
dm_test(ipca_OOB_6meses, previsao_6meses_ARMA, svm_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, previsao_9meses_ARMA, svm_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, previsao_12meses_ARMA, svm_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, previsao_24meses_ARMA, svm_24meses)

# ARMA x LGBM 
dm_test(ipca_OOB_6meses, previsao_6meses_ARMA, lgbm_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, previsao_9meses_ARMA, lgbm_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, previsao_12meses_ARMA, lgbm_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, previsao_24meses_ARMA, lgbm_24meses)



# KNN x SVM
dm_test(ipca_OOB_6meses, knn_6meses, svm_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, knn_9meses, svm_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, knn_12meses, svm_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, knn_24meses,svm_24meses )

# KNN x LGBM
dm_test(ipca_OOB_6meses, knn_6meses, lgbm_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, knn_9meses, lgbm_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, knn_12meses, lgbm_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, knn_24meses, lgbm_24meses )



# SVM x LGBM
dm_test(ipca_OOB_6meses, svm_6meses, lgbm_6meses)
# 9 meses
dm_test(ipca_OOB_9meses, svm_9meses, lgbm_9meses)
# 12 meses
dm_test(ipca_OOB_12meses, svm_12meses, lgbm_12meses)
# 24 meses
dm_test(ipca_OOB_24meses, svm_24meses, lgbm_24meses )



# GRÁFICOS FINAIS
# Primeiro irei plotar gráfico por gráfico. E depois criarei os quatro dentro de uma mesma janela, igual na dissertação
# 6 meses
plt.figure(figsize=(12,5))
plt.plot(np.arange(1,7), ipca_OOB_6meses, label = 'IPCA', color = 'red')
plt.plot(np.arange(1,7), previsao_6meses_AR1, label = 'AR1', color = 'black')
plt.plot(np.arange(1,7), previsao_6meses_ARMA, label = 'ARMA', color = 'purple')
plt.plot(np.arange(1,7), knn_6meses, label = 'KNN', color = 'darkorange')
plt.plot(np.arange(1,7), svm_6meses, label = 'SVM', color = 'blue')
plt.plot(np.arange(1,7), lgbm_6meses, label = 'LGBM', color = 'green')
plt.title('Métodos x IPCA, h = 6')
plt.xlabel('Meses')
plt.ylabel('Tx Inflação')
plt.legend()
plt.show()

# 9 meses
plt.figure(figsize=(12,5))
plt.plot(np.arange(1,10), ipca_OOB_9meses, label = 'IPCA', color = 'red')
plt.plot(np.arange(1,10), previsao_9meses_AR1, label = 'AR1', color = 'black')
plt.plot(np.arange(1,10), previsao_9meses_ARMA, label = 'ARMA', color = 'purple')
plt.plot(np.arange(1,10), knn_9meses, label = 'KNN', color = 'darkorange')
plt.plot(np.arange(1,10), svm_9meses, label = 'SVM', color = 'blue')
plt.plot(np.arange(1,10), lgbm_9meses, label = 'LGBM', color = 'green')
plt.title('Métodos x IPCA, h = 9')
plt.xlabel('Meses')
plt.ylabel('Tx Inflação')
plt.legend()
plt.show()

# 12 meses
plt.figure(figsize=(12,5))
plt.plot(np.arange(1,13), ipca_OOB_12meses, label = 'IPCA', color = 'red')
plt.plot(np.arange(1,13), previsao_12meses_AR1, label = 'AR1', color = 'black')
plt.plot(np.arange(1,13), previsao_12meses_ARMA, label = 'ARMA', color = 'purple')
plt.plot(np.arange(1,13), knn_12meses, label = 'KNN', color = 'darkorange')
plt.plot(np.arange(1,13), svm_12meses, label = 'SVM', color = 'blue')
plt.plot(np.arange(1,13), lgbm_12meses, label = 'LGBM', color = 'green')
plt.title('Métodos x IPCA, h = 12')
plt.xlabel('Meses')
plt.ylabel('Tx Inflação')
plt.legend()
plt.show()

# 24 meses
plt.figure(figsize=(12,5))
plt.plot(np.arange(1,25), ipca_OOB_24meses, label = 'IPCA', color = 'red')
plt.plot(np.arange(1,25), previsao_24meses_AR1, label = 'AR1', color = 'black')
plt.plot(np.arange(1,25), previsao_24meses_ARMA, label = 'ARMA', color = 'purple')
plt.plot(np.arange(1,25), knn_24meses, label = 'KNN', color = 'darkorange')
plt.plot(np.arange(1,25), svm_24meses, label = 'SVM', color = 'blue')
plt.plot(np.arange(1,25), lgbm_24meses, label = 'LGBM', color = 'green')
plt.title('Métodos x IPCA, h = 24')
plt.xlabel('Meses')
plt.ylabel('Tx Inflação')
plt.legend()
plt.show()



## CRIANDO TODOS OS GRÁFICOS DENTRO DE UMA ÚNICA JANELA
fig, axes = plt.subplots(2, 2, figsize = (18,11))

# 6 meses na primeira janela à esquerda
axes[0,0].plot(np.arange(1,7), ipca_OOB_6meses, label = 'IPCA', color = 'red')
axes[0,0].plot(np.arange(1,7), previsao_6meses_AR1, label = 'AR1', color = 'black')
axes[0,0].plot(np.arange(1,7), previsao_6meses_ARMA, label = 'ARMA', color = 'purple')
axes[0,0].plot(np.arange(1,7), knn_6meses, label = 'KNN', color = 'darkorange')
axes[0,0].plot(np.arange(1,7), svm_6meses, label = 'SVM', color = 'blue')
axes[0,0].plot(np.arange(1,7), lgbm_6meses, label = 'LGBM', color = 'green')
axes[0,0].set_title('Métodos x IPCA, h = 6')
axes[0,0].set_xlabel('Meses')
axes[0,0].set_ylabel('Tx Inflação')
axes[0,0].legend(loc = 'best')

# 9 meses na primeira janela à direita
axes[0,1].plot(np.arange(1,10), ipca_OOB_9meses, label = 'IPCA', color = 'red')
axes[0,1].plot(np.arange(1,10), previsao_9meses_AR1, label = 'AR1', color = 'black')
axes[0,1].plot(np.arange(1,10), previsao_9meses_ARMA, label = 'ARMA', color = 'purple')
axes[0,1].plot(np.arange(1,10), knn_9meses, label = 'KNN', color = 'darkorange')
axes[0,1].plot(np.arange(1,10), svm_9meses, label = 'SVM', color = 'blue')
axes[0,1].plot(np.arange(1,10), lgbm_9meses, label = 'LGBM', color = 'green')
axes[0,1].set_title('Métodos x IPCA, h = 9')
axes[0,1].set_xlabel('Meses')
axes[0,1].set_ylabel('Tx Inflação')
axes[0,1].legend(loc = 'best')

# 12 meses
axes[1,0].plot(np.arange(1,13), ipca_OOB_12meses, label = 'IPCA', color = 'red')
axes[1,0].plot(np.arange(1,13), previsao_12meses_AR1, label = 'AR1', color = 'black')
axes[1,0].plot(np.arange(1,13), previsao_12meses_ARMA, label = 'ARMA', color = 'purple')
axes[1,0].plot(np.arange(1,13), knn_12meses, label = 'KNN', color = 'darkorange')
axes[1,0].plot(np.arange(1,13), svm_12meses, label = 'SVM', color = 'blue')
axes[1,0].plot(np.arange(1,13), lgbm_12meses, label = 'LGBM', color = 'green')
axes[1,0].set_title('Métodos x IPCA, h = 12')
axes[1,0].set_xlabel('Meses')
axes[1,0].set_ylabel('Tx Inflação')
axes[1,0].legend(loc = 'best')

# 24 meses
axes[1,1].plot(np.arange(1,25), ipca_OOB_24meses, label = 'IPCA', color = 'red')
axes[1,1].plot(np.arange(1,25), previsao_24meses_AR1, label = 'AR1', color = 'black')
axes[1,1].plot(np.arange(1,25), previsao_24meses_ARMA, label = 'ARMA', color = 'purple')
axes[1,1].plot(np.arange(1,25), knn_24meses, label = 'KNN', color = 'darkorange')
axes[1,1].plot(np.arange(1,25), svm_24meses, label = 'SVM', color = 'blue')
axes[1,1].plot(np.arange(1,25), lgbm_24meses, label = 'LGBM', color = 'green')
axes[1,1].set_title('Métodos x IPCA, h = 24')
axes[1,1].set_xlabel('Meses')
axes[1,1].set_ylabel('Tx Inflação')
axes[1,1].legend(loc = 'best')

# Plotando
fig


















