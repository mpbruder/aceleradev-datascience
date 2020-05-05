#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
dataframe = pd.DataFrame({"normal": sct.norm.rvs(loc=20, scale=4, size=10000),
                          "binomial": sct.binom.rvs(100, 0.2, size=10000)}) # loc é a média, scale é o desvio padrão.


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[301]:


# Sua análise da parte 1 começa aqui.
dataframe['normal'].describe()


# In[302]:


dataframe['binomial'].describe()


# In[303]:


sns.distplot(dataframe['normal'])


# ## Ex01

# In[304]:


# Quartis da da distribuição normal
q1_norm = dataframe['normal'].quantile(0.25)
q2_norm = dataframe['normal'].quantile(0.5)
q3_norm = dataframe['normal'].quantile(0.75)


# In[305]:


# Quartis da distribuicao binomial
q1_binom = dataframe['binomial'].quantile(0.25)
q2_binom = dataframe['binomial'].quantile(0.5)
q3_binom = dataframe['binomial'].quantile(0.75)


# In[306]:


# Operacoes
r1 = round(q1_norm - q1_binom, 3)
r2 = round(q2_norm - q2_binom, 3)
r3 = round(q3_norm - q3_binom, 3)


# In[307]:


resp = (r1, r2, r3)


# ## Ex02

# In[308]:


# Ponto minimo e maximo do intervalo
min_interv = dataframe['normal'].mean() - dataframe['normal'].std()
max_interv = dataframe['normal'].mean() + dataframe['normal'].std()


# In[309]:


# Calculo do CDF max e min
max_cdf = sct.norm.cdf(max_interv, loc=20, scale=4)
min_cdf = sct.norm.cdf(min_interv, loc=20, scale=4)


# In[310]:


# Calculo do ECDF
my_ecdf = round(max_cdf - min_cdf, 3)


# In[311]:


my_ecdf


# ## Ex03
# 

# In[312]:


# Media e variancia da normal
m_norm = dataframe['normal'].mean()
v_norm = dataframe['normal'].var()


# In[313]:


# Media e variancia da binomial
m_binom = dataframe['binomial'].mean()
v_binom = dataframe['binomial'].var()


# In[314]:


# calculo diferenca
r1 = round(m_binom - m_norm, 3)
r2 = round(v_binom - v_norm, 3)


# In[315]:


resp = (r1, r2)


# In[316]:


resp


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[317]:


def q1():
    # Retorne aqui o resultado da questão 1.
    # Quartis da da distribuição normal
    q1_norm = dataframe['normal'].quantile(0.25)
    q2_norm = dataframe['normal'].quantile(0.5)
    q3_norm = dataframe['normal'].quantile(0.75)
    #Quartis da distribuicao binomial
    q1_binom = dataframe['binomial'].quantile(0.25)
    q2_binom = dataframe['binomial'].quantile(0.5)
    q3_binom = dataframe['binomial'].quantile(0.75)
    # Operacoes
    r1 = round(q1_norm - q1_binom, 3)
    r2 = round(q2_norm - q2_binom, 3)
    r3 = round(q3_norm - q3_binom, 3)
    tupla = (r1, r2, r3)
    return tupla 
    pass


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[318]:


def q2():
    # Retorne aqui o resultado da questão 2.
    # Ponto minimo e maximo do intervalo
    min_interv = dataframe['normal'].mean() - dataframe['normal'].std()
    max_interv = dataframe['normal'].mean() + dataframe['normal'].std()
    # Calculo do CDF max e min
    max_cdf = sct.norm.cdf(max_interv, loc=20, scale=4)
    min_cdf = sct.norm.cdf(min_interv, loc=20, scale=4)
    # Calculo do ECDF
    my_ecdf = round(max_cdf - min_cdf, 3) 
    return float (my_ecdf) 
    pass


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[319]:


def q3():
    # Retorne aqui o resultado da questão 3.
    # Media e variancia da normal
    m_norm = dataframe['normal'].mean()
    v_norm = dataframe['normal'].var()
    # Media e variancia da binomial
    m_binom = dataframe['binomial'].mean()
    v_binom = dataframe['binomial'].var()
    # calculo diferenca
    r1 = round(m_binom - m_norm, 3)
    r2 = round(v_binom - v_norm, 3)
    tupla = (r1, r2)
    return tupla
    pass


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[4]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# ## Ex04

# In[23]:


stars.head()


# In[37]:


filtro = stars[stars['target'] == 0]


# In[39]:


false_pulsar_mean_profile = filtro['mean_profile']


# In[43]:


false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - false_pulsar_mean_profile.mean()) / false_pulsar_mean_profile.std()


# In[44]:


# Quantis teoricos
i1, i2, i3 = sct.norm.ppf([0.8, 0.9, 0.95])


# In[45]:


r1, r2, r3 = ECDF(false_pulsar_mean_profile_standardized)([i1, i2, i3])


# In[46]:


tupla = (round(r1, 3), round(r2, 3), round(r3, 3))


# In[47]:


tupla


# ## Ex05
# 

# In[363]:


# Quantis de 'false_pulsar_mean_profile_standardized'
q1_fpmps = false_pulsar_mean_profile_standardized.quantile(0.25)
q2_fpmps = false_pulsar_mean_profile_standardized.quantile(0.5)
q3_fpmps = false_pulsar_mean_profile_standardized.quantile(0.75)


# In[364]:


# Quartis de distribuicao normal com media igual a 0 e variancia igual a 1
q1_distnorm = sct.norm.ppf(0.25, loc=0, scale=1)
q2_distnorm = sct.norm.ppf(0.5, loc=0, scale=1)
q3_distnorm = sct.norm.ppf(0.75, loc=0, scale=1)


# In[365]:


# Diferencas entre quartis
r1 = round(q1_fpmps - q1_distnorm, 3)
r2 = round(q2_fpmps - q2_distnorm, 3)
r3 = round(q3_fpmps - q3_distnorm, 3)


# In[366]:


tupla = (r1, r2, r3)


# In[367]:


tupla


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[359]:


def q4():
    # Retorne aqui o resultado da questão 4.
    filtro = stars[stars['target'] == 0]
    false_pulsar_mean_profile = filtro['mean_profile']
    false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - false_pulsar_mean_profile.mean()) / false_pulsar_mean_profile.std()
    # Quantis teoricos
    i1, i2, i3 = sct.norm.ppf([0.8, 0.9, 0.95])
    r1, r2, r3 = ECDF(false_pulsar_mean_profile_standardized)([i1, i2, i3])
    tupla = (round(r1, 3), round(r2, 3), round(r3, 3))
    return tupla
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[360]:


def q5():
    # Quantis de 'false_pulsar_mean_profile_standardized'
    q1_fpmps = false_pulsar_mean_profile_standardized.quantile(0.25)
    q2_fpmps = false_pulsar_mean_profile_standardized.quantile(0.5)
    q3_fpmps = false_pulsar_mean_profile_standardized.quantile(0.75)
    # Quartis de distribuicao normal com media igual a 0 e variancia igual a 1
    q1_distnorm = sct.norm.ppf(0.25, loc=0, scale=1)
    q2_distnorm = sct.norm.ppf(0.5, loc=0, scale=1)
    q3_distnorm = sct.norm.ppf(0.75, loc=0, scale=1)
    # Diferencas entre quartis
    r1 = round(q1_fpmps - q1_distnorm, 3)
    r2 = round(q2_fpmps - q2_distnorm, 3)
    r3 = round(q3_fpmps - q3_distnorm, 3)
    tupla = (r1, r2, r3)
    return tupla
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
