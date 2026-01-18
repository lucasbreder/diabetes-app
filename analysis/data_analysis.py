import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import styles
import seaborn as sns

styles.apply_styles()
df = pd.read_csv("./dataset/diabetes.csv")

for column in df:
    zero_count = (df[column] == 0).sum()
    print(f"Coluna '{column}': {zero_count} ocorrências de 0")

### Verificação de tipagem dos dados
print("-" * 50)
print("Verificação de tipagem dos dados")
print("-" * 50)
print(df.info())

### Verificação de dados nulos
print("-" * 50)
print("Verificação de dados nulos")
print("-" * 50)
print(df.isnull().sum())

### Verificação de dados duplicados
print("-" * 50)
print("Verificação de dados duplicados")
print("-" * 50)
print(df.duplicated().sum())

### Verificação de correlação maior que 0.4
print("-" * 50)
print("Verificação de correlação maior que 0.4")
print("-" * 50)
print(df.corr())

##Verificação de distribuição
print("-" * 50)
print("Verificação de distribuição")
print("-" * 50)
print(df.describe())

### Histogramas de distribuição

df.hist(bins=20, figsize=(20,15), color='skyblue')
print("--- Gerando gráfico...")
plt.savefig('./graphs/histogram_distribution.png')
print("--- Gráfico salvo como histogram_distribution.png")

### Distribuição de Idades

fig, ax = plt.subplots(figsize=(8, 5))

age_mean = df["Age"].mean()
age_std = df["Age"].std()
age_med = df["Age"].median()

ax.hist(df["Age"].dropna(), bins=20, alpha=0.8, color='skyblue', zorder=3)

inf_limit = age_mean - age_std
sup_limit = age_mean + age_std

total_inside_1_sigma =  df["Age"].between(inf_limit, sup_limit).sum()
age_total = len( df["Age"])
age_total_1_sigma_percent = (total_inside_1_sigma / age_total) * 100

ax.set_title("Distribuição de Idade", pad=30)
ax.set_xlabel("Idade", labelpad=20)
ax.set_ylabel("Frequência", labelpad=20)
ax.grid(visible=True, axis='y', linestyle='--', alpha=0.7)
ax.axvline(
    x=age_mean,
    color='#AA6D43',
    linestyle='--',
    linewidth=1,
    label=f'Média: {age_mean:.1f} anos',
    zorder=4
)
ax.axvspan(
    xmin=inf_limit,
    xmax=sup_limit,
    alpha=0.2,
    color='gray',
    label=f'Dentro de 1 sigma: {age_total_1_sigma_percent:.1f}%',
    zorder=2
)
ax.axvline(
    x=age_med,
    color='#FF710F',
    linestyle='--',
    linewidth=1,
    label=f'Mediana: {age_med:.1f} anos',
    zorder=4
)
ax.legend()

print("--- Gerando gráfico...")
plt.savefig('./graphs/histogram_age.png')
print("--- Gráfico salvo como histogram_age.png")

### Relação entre diagnóstico de diabetes e medição da glicose

count, borders = np.histogram(df.loc[df["Outcome"] == 1, "Glucose"].dropna(), bins=20)
bins_centers = 0.5 * (borders[1:] + borders[:-1])

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
   bins_centers,
    count,
    color='#4d4d4d',
    linestyle='-',
    linewidth=2,
    marker='o',
    label='Diabética',
   zorder=4
)
ax.hist(df.loc[df["Outcome"] == 0, "Glucose"].dropna(), bins=20, alpha=1, color='skyblue', zorder=3, label='Não Diabética')
ax.grid(visible=True, axis='y', linestyle='--', alpha=0.7)
plt.title("Distribuição de Glicose por Diagnóstico")
plt.xlabel("Glicose")
plt.ylabel("Frequência")
ax.axvline(
    x=140,
    color='#FF710F',
    linestyle='--',
    label='140',
    linewidth=1,
    zorder=4
)
ax.axvline(
    x=200,
    color='#55473E',
    linestyle='--',
    linewidth=1,
    label='200',
    zorder=4
)
ax.legend()

print("--- Gerando gráfico...")
plt.savefig('./graphs/histogram_glucose.png')
print("--- Gráfico salvo como histogram_glucose.png")

### Relação entre espessura da pele e idade

import matplotlib.pyplot as plt

dados_limpos = df[df["SkinThickness"] > 0]
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    dados_limpos["Age"],
    dados_limpos["SkinThickness"],
    alpha=0.2,
    color='red',
    edgecolor='white',
    s=100
)
ax.scatter(
    x=[24.5],
    y=[19.5],
    color='gold',
    s=100,
    marker='o',
    edgecolor='black',
    zorder=10,
    alpha=1,
    label='Média (20-29 anos) 1990',
)
ax.scatter(
    x=[34.5],
    y=[22.5],
    color='green',
    s=100,
    marker='o',
    edgecolor='black',
    zorder=10,
    alpha=1,
    label='Média (30-39 anos) 1990'
)
ax.scatter(
    x=[44.5],
    y=[24.5],
    color='blue',
    s=100,
    marker='o',
    edgecolor='black',
    zorder=10,
    alpha=1,
    label='Média (40-49 anos) 1990'
)


ax.set_title("Relação: Idade vs Espessura da Pele (Gordura Subcutânea)")
ax.set_xlabel("Idade (anos)")
ax.set_ylabel("Espessura da Pele (mm)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

print("--- Gerando gráfico...")
plt.savefig('./graphs/age_skin_thickness.png')
print("--- Gráfico salvo como age_skin_thickness.png")

### Frequência de DPF por Idade

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

sns.regplot(
    data=dados_limpos,
    x="Age",
    y="DiabetesPedigreeFunction",
    ax=ax,
    color='red',
    scatter_kws={'alpha':0.2, 's':100, 'edgecolor':'white'}, # Estilo das bolinhas
    line_kws={'color':'blue'} # Estilo da linha
)

ax.set_title("Relação: DPF vs Idade (Com Regressão)")
plt.grid(True, linestyle='--', alpha=0.5)
print("--- Gerando gráfico...")
plt.savefig('./graphs/dpf_age.png')
print("--- Gráfico salvo como dpf_age.png")
