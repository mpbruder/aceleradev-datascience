#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, KBinsDiscretizer, StandardScaler)


# In[3]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


countries = pd.read_csv("countries.csv", thousands='.', decimal=',')


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# In[6]:


countries.shape


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.


# In[8]:


df = pd.DataFrame({ 'colunas' : countries.columns,
                    'tipos' : countries.dtypes,
                    'missing' : countries.isna().sum(),
                    'size' : countries.shape[0],
                    'unicos' : countries.nunique()}
                 )
df['missing_perc'] = (df['missing'] / df['size']).round(2)


# In[9]:


df


# In[10]:


# Maximo é 10% de valor faltante, creio que pode-se dropar NAs
#df_countries = countries.dropna()


# In[11]:


# Copiar dataframe para nao fazer cagada
df_countries = countries.copy()


# In[12]:


# Transformei direto da importação com: "pd.read_csv("countries.csv", thousands='.', decimal=',')"
df_countries.dtypes


# In[13]:


# Remover espaços com strip
df_countries['Country'] = df_countries['Country'].str.strip()
df_countries['Region'] = df_countries['Region'].str.strip()


# In[14]:


# Strip funcionou?
#countries_notna['Country'][0], countries['Country'][0]
df_countries['Region'][0], countries['Region'][0]


# In[15]:


df_countries.head(5)


# In[16]:


df_countries.shape


# ### Exercicio 01

# In[17]:


region_uni = df_countries['Region'].unique()


# In[18]:


sorted(region_uni)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[19]:


def q1():
    # Retorne aqui o resultado da questão 1.
    #Selecionar apenas os valores unicos de "countries['Region']"
    region_uni = df_countries['Region'].unique()
    return sorted(region_uni)


# ### Exercicio 02

# In[20]:


def get_interval(bin_idx, bin_edges):
  return f"{np.round(bin_edges[bin_idx], 2):.2f} ⊢ {np.round(bin_edges[bin_idx+1], 2):.2f}"


# In[21]:


# criando objeto discretizer com 10 intervalos (bins), usando estrategia 'quantile'
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")


# In[22]:


# treinando modelo 
discretizer.fit(df_countries[['Pop_density']])


# In[23]:


# gerando limites de intervalos
score_bins = discretizer.transform(df_countries[['Pop_density']])


# In[24]:


# visualizando os intervalos gerados
discretizer.bin_edges_


# In[25]:


# confirmando numero de intervalos gerados
len(discretizer.bin_edges_[0])-1


# In[26]:


#somando apenas os dados que estão acima do 90o percentilsub
sum(score_bins[:, 0] == 9)


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[27]:


def q2():
    # Retorne aqui o resultado da questão 2.

    # criando objeto discretizer com 10 intervalos (bins), usando estrategia 'quantile'
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    # treinando modelo 
    discretizer.fit(df_countries[['Pop_density']])
    # gerando limites de intervalos
    score_bins = discretizer.transform(df_countries[['Pop_density']])
    # visualizando os intervalos gerados
    discretizer.bin_edges_
    #somando apenas os dados que estão acima do 90o percentilsub
    return int (sum(score_bins[:, 0] == 9))


# ### Exercicio 03

# In[28]:


# criar objeto de OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int64)


# In[29]:


# Preenchendo valores nulos de 'Climate'
df_countries['Climate'].fillna(0, inplace=True)


# In[30]:


# Treinar e transformar (encodar) os atributos 'Region' e 'Climate'
encoded = one_hot_encoder.fit_transform(df_countries[['Region','Climate']])


# In[31]:


# atributos criados durante o treino e transformacao
new_atributes = encoded.shape[1]


# In[32]:


new_atributes


# ## Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[33]:


def q3():
    # Retorne aqui o resultado da questão 3.

    # criar objeto de OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    # Preenchendo valores nulos de 'Climate'
    df_countries['Climate'].fillna(0, inplace=True)
    # Treinar e transformar (encodar) os atributos 'Region' e 'Climate'
    encoded = one_hot_encoder.fit_transform(df_countries[['Region','Climate']])
    # atributos criados durante o treino e transformacao
    new_atributes = encoded.shape[1]
    return int(new_atributes)


# ### Exercicio 04

# In[34]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[35]:


# criar pipeline
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())
])


# In[36]:


# selecionar apenas atributos numericos
numeric_atributes = countries.describe().columns.tolist()


# In[37]:


# aplicar pipeline sobre dados de treino (fit & transform) | retorna lista
pipeline_transformation = num_pipeline.fit_transform(df_countries[numeric_atributes])


# In[38]:


# Transformar lista em um DataFrame e testar
df_test_country = pd.DataFrame([test_country], columns = df_countries.columns)

df_test_country.head()


# In[39]:


# Aplicar pipeline sobre o df, apenas nas colunas numericas
test_country_transformation = num_pipeline.transform(df_test_country[numeric_atributes])


# In[40]:


# Criando df dos dados transformados para pegar apenas 'Arable'
df_test_country_pipeline = pd.DataFrame(test_country_transformation, columns = df_countries.select_dtypes(include=[np.number]).columns)


# In[41]:


df_test_country_pipeline['Arable'].round(3)


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[42]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[43]:


def q4():
    # Retorne aqui o resultado da questão 4.
    
    # criar pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("standard_scaler", StandardScaler())
    ])
    # selecionar apenas atributos numericos
    numeric_atributes = countries.describe().columns.tolist()
    # aplicar pipeline sobre dados de treino (fit & transform) | retorna lista
    pipeline_transformation = num_pipeline.fit_transform(df_countries[numeric_atributes])
    # Transformar lista em um DataFrame e testar
    df_test_country = pd.DataFrame([test_country], columns = df_countries.columns)
    df_test_country.head()
    # Aplicar pipeline sobre o df, apenas nas colunas numericas
    test_country_transformation = num_pipeline.transform(df_test_country[numeric_atributes])
    # Criando df dos dados transformados para pegar apenas 'Arable'
    df_test_country_pipeline = pd.DataFrame(test_country_transformation, columns = df_countries.select_dtypes(include=[np.number]).columns)
    return float(df_test_country_pipeline['Arable'].round(3))


# ### Exercicio 05

# In[98]:


# selecionar apenas colunas net_migration
net_migration_outliers = df_countries['Net_migration'].dropna().copy()


# In[99]:


# plotando grafico boxplot da variavel
sns.boxplot(net_migration_outliers, orient="vertical");


# In[100]:


# Descobrindo o intervalo em que os dados são normais
quantil1 = net_migration_outliers.quantile(0.25)
quantil3 = net_migration_outliers.quantile(0.75)
iqr = quantil3 - quantil1

non_outlier_interval_iqr = [(quantil1 - 1.5 * iqr), (quantil3 + 1.5 * iqr)]


# In[101]:


non_outlier_interval_iqr


# In[103]:


# Descobrindo outliers do atributo 'net_migration'
amount_lower_outliers = net_migration_outliers[(net_migration_outliers < non_outlier_interval_iqr[0]).sum()]
outliers_abaixo = net_migration_outliers[(net_migration_outliers < non_outlier_interval_iqr[0])]

amount_higher_outliers = net_migration_outliers[(net_migration_outliers > non_outlier_interval_iqr[1]).sum()]
outliers_acima = net_migration_outliers[(net_migration_outliers > non_outlier_interval_iqr[1])]


# In[104]:


len(outliers_abaixo), len(net_migration_outliers)


# In[105]:


# devo desconsiderar os outliers?
drop_outliers = bool((amount_lower_outliers/len(net_migration_outliers)) > .05 or ((amount_higher_outliers/len(net_migration_outliers)) > .05))


# In[107]:


len(outliers_abaixo), len(outliers_acima), drop_outliers


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[109]:


def q5():
    # Retorne aqui o resultado da questão 5.

    # selecionar apenas colunas net_migration
    net_migration_outliers = df_countries['Net_migration'].dropna().copy()
    # plotando grafico boxplot da variavel
    sns.boxplot(net_migration_outliers, orient="vertical");
    # Descobrindo o intervalo em que os dados são normais
    quantil1 = net_migration_outliers.quantile(0.25)
    quantil3 = net_migration_outliers.quantile(0.75)
    iqr = quantil3 - quantil1

    non_outlier_interval_iqr = [(quantil1 - 1.5 * iqr), (quantil3 + 1.5 * iqr)]
    # Descobrindo outliers do atributo 'net_migration'
    amount_lower_outliers = net_migration_outliers[(net_migration_outliers < non_outlier_interval_iqr[0]).sum()]
    outliers_abaixo = net_migration_outliers[(net_migration_outliers < non_outlier_interval_iqr[0])]

    amount_higher_outliers = net_migration_outliers[(net_migration_outliers > non_outlier_interval_iqr[1]).sum()]
    outliers_acima = net_migration_outliers[(net_migration_outliers > non_outlier_interval_iqr[1])]
    # devo desconsiderar os outliers?
    drop_outliers = bool((amount_lower_outliers/len(net_migration_outliers)) > .05 or ((amount_higher_outliers/len(net_migration_outliers)) > .05))
    return len(outliers_abaixo), len(outliers_acima), drop_outliers


# ### Exercicio 06

# In[52]:


# carregando categorias e o dataset
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[53]:


# documento total
len(newsgroups.data)


# In[54]:


# transformacao por contagem de palavras - criando obj. de CountVectorizer()
count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
type(newsgroups_counts)


# In[55]:


# selecionando todas palavras 
words = pd.DataFrame(newsgroups_counts.toarray(), columns=count_vectorizer.get_feature_names())


# In[56]:


# somando apenas 'phones'
words['phone'].sum()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[57]:


def q6():
    # Retorne aqui o resultado da questão 6.

    # carregando categorias e o dataset
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    # transformacao por contagem de palavras - criando obj. de CountVectorizer()
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    type(newsgroups_counts)
    # selecionando todas palavras 
    words = pd.DataFrame(newsgroups_counts.toarray(), columns=count_vectorizer.get_feature_names())
    return int(words['phone'].sum())


# ### Exercicio 07

# In[58]:


# criando objeto de TfidfVectorizer()
tfidf_vectorizer = TfidfVectorizer()


# In[59]:


# treinando o modelo
newsgroups_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups.data)


# In[60]:


# criando uma dataframe para selecionar dado
newsgroups_tfidf = pd.DataFrame(newsgroups_tfidf_vectorized.toarray(), columns=tfidf_vectorizer.get_feature_names())


# In[61]:


newsgroups_tfidf['phone'].sum().round(3)


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[62]:


def q7():
    # Retorne aqui o resultado da questão 7.
    
    # criando objeto de TfidfVectorizer()
    tfidf_vectorizer = TfidfVectorizer()
    # treinando o modelo
    newsgroups_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups.data)
    # criando uma dataframe para selecionar dado
    newsgroups_tfidf = pd.DataFrame(newsgroups_tfidf_vectorized.toarray(), columns=tfidf_vectorizer.get_feature_names())
    return float(newsgroups_tfidf['phone'].sum().round(3))


# In[ ]:




