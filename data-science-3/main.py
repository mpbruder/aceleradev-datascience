#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[93]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
import sklearn as sk

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from loguru import logger


# In[40]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[41]:


fifa = pd.read_csv("fifa.csv")


# In[42]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[43]:


# Sua análise começa aqui.


# In[44]:


fifa.shape


# In[45]:


fifa.head()


# In[46]:


#Criando um dataframe auxliar para analisar a consistencia das variaveis
df = pd.DataFrame({'colunas' : fifa.columns,
                    'tipo': fifa.dtypes,
                    'missing' : fifa.isna().sum(),
                    'size' : fifa.shape[0],
                    'unicos': fifa.nunique()})
df['percentual'] = round(df['missing'] / df['size'],2)


# In[49]:


df


# In[47]:


# Dropar valores nulos pois são poucos!
fifa.dropna(inplace=True)


# In[48]:


fifa.shape


# ## Exercicio 01

# In[50]:


#instanciando PCA na variável pca.
pca = PCA()


# In[51]:


# Treinando meu dataframe
pca.fit(fifa)


# In[53]:


evr = pca.explained_variance_ratio_


# In[55]:


evr


# In[56]:


evr[0]


# ## Exercicio 02

# In[57]:


cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
component_number = np.argmax(cumulative_variance_ratio >= 0.95) + 1 # Contagem começa em zero.

component_number


# In[58]:


evr_acumulado = np.cumsum(evr)


# In[59]:


num_features = np.argmax(evr_acumulado >= 0.95) + 1 # Contagem começa em zero.


# In[60]:


num_features


# ## Exercicio 03

# In[110]:


# definindo objeto PCA com PC1 e PC2
pca_componentes = PCA(n_components=2)


# In[111]:


#passo os dados (dataframe) para o objeto de pca para reduzir
pca_componentes.fit(fifa)


# In[112]:


#PC = pca_componentes.components_.dot(x)


# In[113]:


#tuple(PC.round(3))


# ## Exercicio 04

# In[85]:


#Separando a variavel target e as labels
target = fifa['Overall']
label = fifa.drop(['Overall'],axis=1)


# In[90]:


# Criando um modelo de regressao linear
lr = LinearRegression()


# In[94]:


# Criando um RFE para selecionar as n_features melhores para o modelo
selector = RFE(lr, n_features_to_select=5, step=1)


# In[95]:


# Fittando o modelo com os dados de X,y
selector = selector.fit(label, target)


# In[97]:


selector.support_


# In[99]:


selector.ranking_


# In[102]:


list(label.columns[selector.support_])


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[65]:


def q1():
    # Retorne aqui o resultado da questão 1.

    # instanciando PCA na variável pca.
    pca = PCA() 
    # passo os dados (dataframe) para o objeto de pca para reduzir
    pca.fit(fifa)
    # extraindo variancia explicada
    evr = pca.explained_variance_ratio_
    return float (round(evr[0],3))


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[66]:


def q2():
    # Retorne aqui o resultado da questão 2.
    
    # Descobrindo o EVR acumulado
    evr_acumulado = np.cumsum(evr)
    # Qual o máximo acumulado antes ou igual a 0.95 (95% da variancia total)
    num_features = np.argmax(evr_acumulado >= 0.95) + 1 # Contagem começa em zero, portanto -> +1.
    return int (num_features)


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[84]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[83]:


def q3():
    # Retorne aqui o resultado da questão 3.

    # definindo objeto PCA com PC1 e PC2
    pca_components = PCA(n_components=2)
    # passo os dados (dataframe) para o objeto de pca para reduzir
    pca_components.fit(fifa)
    # componentes principais
    PCs = pca_components.components_.dot(x)
    return tuple(PCs.round(3))


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[82]:


def q4():
    # Retorne aqui o resultado da questão 4.

    #Separando a variavel target e as labels
    target = fifa['Overall']
    label = fifa.drop(['Overall'],axis=1)
    # Criando um modelo de regressao linear
    lr = LinearRegression()
    # Criando um RFE para selecionar as n_features melhores para o modelo
    selector = RFE(lr, n_features_to_select=5, step=1)
    # Fittando o modelo com os dados de X,y
    selector = selector.fit(label, target)
    return list(label.columns[selector.support_])

