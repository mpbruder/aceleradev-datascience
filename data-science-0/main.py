#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# #### Testes Iniciais

# In[4]:


dataframe = pd.DataFrame(black_friday)


# In[5]:


dataframe.head(1)


# In[6]:


dataframe.info()


# In[7]:


dataframe.columns


# #### Ex01

# In[10]:


dataframe.shape


# #### Ex02

# In[11]:


dfFILTRADO = dataframe[(dataframe['Age']=='26-35') & (dataframe['Gender']=='F')]


# In[12]:


len(dfFILTRADO)


# #### Ex03

# In[18]:


df_norepeat  = dataframe['User_ID']


# In[21]:


tot = df_norepeat.nunique()


# In[22]:


tot


# #### Ex04

# In[26]:


columnsType = pd.DataFrame({'colunas': dataframe.columns,
                            'tipos': dataframe.dtypes})


# In[29]:


columnsType['tipos'].nunique()


# #### Ex05

# In[56]:


df_nonulls = dataframe.dropna()


# In[57]:


nRows_Nonulls = df_nonulls.shape[0]


# In[58]:


nRows = dataframe.shape[0]


# In[60]:


percent = 1 - (nRows_Nonulls / nRows)


# In[61]:


percent


# #### Ex06
# 

# In[65]:


col_nulls = dataframe.isnull().sum()


# In[68]:


col_nulls


# #### Ex07
# 

# In[146]:


moda = dataframe['Product_Category_3'].mode()


# In[147]:


float (moda)


# #### Ex08

# In[10]:


media = dataframe['Purchase'].mean()


# In[11]:


media


# In[13]:


df = dataframe['Purchase']


# In[15]:


norm_df = (df - df.min()) / (df.max() - df.min())


# In[16]:


norm_df.mean()


# #### Ex09
# 

# In[8]:


df = dataframe['Purchase']


# In[9]:


pad_df = (df - df.mean()) / df.std()


# In[13]:


mask01 = pad_df > -1


# In[15]:


mask02 = pad_df < 1


# In[17]:


pad_df[mask01 & mask02].count()


# #### Ex 10

# In[125]:


df = dataframe[['Product_Category_2', 'Product_Category_3']]


# In[126]:


mask = df.isna()


# In[127]:


resp = mask[(mask['Product_Category_2']==False) & (mask['Product_Category_2']==True)]


# In[130]:


resp.shape[0]


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[1]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return dataframe.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    mask01 = dataframe['Age']=='26-35'
    mask02 = dataframe['Gender']=='F'
    df_filt = dataframe[mask01 & mask02]
    return len(df_filt)
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    df_norepeat = dataframe['User_ID']
    tot = df_norepeat.nunique()
    return int (tot)
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[78]:


def q4():
    # Retorne aqui o resultado da questão 4.
    columnsType = pd.DataFrame({'colunas': dataframe.columns,
                                'tipos': dataframe.dtypes})
    tot = columnsType['tipos'].nunique()
    return int (tot)
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    df_nonulls = dataframe.dropna()
    nRows_Nonulls = df_nonulls.shape[0]
    nRows = dataframe.shape[0]
    percent = 1 - (nRows_Nonulls / nRows)
    return float (percent)
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    col_nulls = dataframe.isnull().sum()
    col_maxNulls = col_nulls.max()
    return int (col_maxNulls)
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    df_aux = dataframe['Product_Category_3']
    moda = df_aux.dropna().mode()
    return float (moda)
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    # Retorne aqui o resultado da questão 8.
    df = dataframe['Purchase']
    norm_df = (df - df.min()) / (df.max() - df.min())
    media_norm = norm_df.mean()
    return float (media_norm)
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[22]:


def q9():
    # Retorne aqui o resultado da questão 9.
    df = dataframe['Purchase']
    pad_df = (df - df.mean()) / df.std()
    mask01 = pad_df > -1
    mask02 = pad_df < 1
    tot = pad_df[mask01 & mask02].count()
    return int (tot)
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[92]:


def q10():
    # Retorne aqui o resultado da questão 10.
    df = dataframe[['Product_Category_2', 'Product_Category_3']]
    mask = df.isna()
    filtro = mask[(mask['Product_Category_2']==False) & (mask['Product_Category_2']==True)]
    if filtro.shape[0] == 0:
        return True
    else:
        return False
    pass

