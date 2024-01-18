# Databricks notebook source
# Importando bibliotecas

from pyspark.sql import SparkSession
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# Carregando dados
dados = spark.read.csv('/FileStore/tables/Restaurant_revenue__1_-1.csv', sep=',', header=True, inferSchema=True)
df = dados.toPandas()

# COMMAND ----------

display(df)

# COMMAND ----------

df.isna().sum()

# COMMAND ----------

# Codificando dados categoricos
encoder = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['Cuisine_Type']), remainder='passthrough', sparse_threshold=False)
df = encoder.fit_transform(df)
pd.DataFrame(df)

# COMMAND ----------

# Separando dados target e previsão
previsores = df[:, 0:10]
target = df[:, 10:]
target = np.asarray(target)

# COMMAND ----------

# Normalizando previsores
normalizador = MinMaxScaler()
previsores = normalizador.fit_transform(previsores)

# COMMAND ----------

previsores.shape

# COMMAND ----------

# Estruturando target
target = target.reshape(target.shape[0],)
target.shape

# COMMAND ----------

# Reservando dados para treino e teste
previsores_treino, previsores_teste, target_treino, target_teste = train_test_split(previsores, target, test_size=0.1)

# COMMAND ----------

# Treinando modelo
modelo = HistGradientBoostingRegressor(max_iter=150)
resultado = cross_val_score(estimator=modelo, X=previsores_treino, y=target_treino, cv=10, scoring='neg_mean_absolute_error')
modelo.fit(previsores, target)

# COMMAND ----------

# Visualizando resultados do treinamento
media = resultado.mean()
desvio = resultado.std()

plt.figure(figsize=(16,8))
plt.plot(resultado)
plt.title('Resultado do Treinamento\n'+'Média:'+str(media)+'\nDesvio Padrão:'+str(desvio))
plt.xlabel('épocas')
plt.ylabel('Perda')

# COMMAND ----------

# Realizando previsões com dados de teste
x = modelo.predict(previsores_teste)

plt.figure(figsize=(16,8))
plt.plot(x, color='red')
plt.plot(target_teste)
plt.title('Previsão de receita mensal')
plt.xlabel('épocas')
plt.ylabel('valor')
plt.legend(['Previsão','Valor Real'])
