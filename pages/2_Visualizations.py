import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Визуализации", layout="wide")
st.title("Графический анализ данных")

df = pd.read_csv("energyoutput.csv")

# Удаляем служебные столбцы и столбцы с датой/временем
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Удаляем столбцы, которые содержат строки (дата/время)
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col])
            df.drop(col, axis=1, inplace=True)
        except:
            pass

# Также удаляем явно известные временные столбцы
for col in ['date', 'UTC', 'Data', 'timestamp']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# 1. Гистограмма целевой переменной
fig1, ax1 = plt.subplots(figsize=(8, 4))   # уменьшенный размер
sns.histplot(df['Appliances'], bins=50, kde=True, ax=ax1)
ax1.set_title("Распределение энергопотребления")
st.pyplot(fig1)

# 2. Корреляционная матрица (первые 10 признаков, исключая Appliances)
numeric_cols = df.select_dtypes(include=['number']).columns
corr_data = df[numeric_cols].drop('Appliances', axis=1).iloc[:, :10]
fig2, ax2 = plt.subplots(figsize=(8, 6))   # уменьшенный размер
sns.heatmap(corr_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# 3. Температура vs энергопотребление
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=df, x='T_out', y='Appliances', alpha=0.5, ax=ax3)
ax3.set_title("Энергопотребление vs Температура на улице")
st.pyplot(fig3)

# 4. Боксплот: влажность по группам энергопотребления
df['Appliances_group'] = pd.cut(df['Appliances'], bins=5)
fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x='Appliances_group', y='RH_out', ax=ax4)
plt.xticks(rotation=45)
ax4.set_title("Уличная влажность по группам энергопотребления")
st.pyplot(fig4)