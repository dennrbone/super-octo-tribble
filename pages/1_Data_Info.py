import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="О датасете", layout="wide")
st.title("Информация о наборе данных")

try:
    df = pd.read_csv("energyoutput.csv")
except FileNotFoundError:
    st.error("Файл energyoutput.csv не найден. Убедитесь, что он находится в той же директории, что и приложение.")
    st.stop()

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

st.markdown("""
### Предметная область
**Прогнозирование энергопотребления бытовых приборов**  
Датасет содержит измерения различных датчиков (температура, влажность, давление, качество воздуха и т.д.) в жилом доме.  
Целевая переменная — `Appliances` — энергопотребление в ватт-часах (Вт·ч).
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("Количество наблюдений", f"{df.shape[0]:,}")
with col2:
    st.metric("Количество признаков", df.shape[1] - 1)

st.subheader("Признаки датасета")

feature_info = pd.DataFrame({
    "Признак": df.columns,
    "Тип данных": df.dtypes.astype(str),
    "Кол-во уникальных": [df[col].nunique() for col in df.columns],
    "Процент пропусков": [round(df[col].isna().mean() * 100, 2) for col in df.columns]
})
st.dataframe(feature_info, use_container_width=True)

st.markdown("""
**Описание основных признаков:**

| Признак | Описание |
|---------|----------|
| `Appliances` | Энергопотребление (целевая переменная), Вт·ч |
| `lights` | Энергопотребление освещения, Вт·ч |
| `T1`–`T9` | Температура в 9 различных зонах дома, °C |
| `RH_1`–`RH_9` | Относительная влажность в соответствующих зонах, % |
| `T_out` | Температура на улице, °C |
| `Press_mm_hg` | Атмосферное давление, мм рт. ст. |
| `RH_out` | Относительная влажность на улице, % |
| `Windspeed` | Скорость ветра, м/с |
| `Visibility` | Видимость, км |
| `Tdewpoint` | Температура точки росы, °C |
| `Year`, `Month`, `Day_of_week`, `Hour`, `Minut` | Временные признаки, извлечённые из столбца `date` |
| `Weekend`, `Night`, `Morning`, `Afternoon`, `Evening` | Бинарные признаки времени суток и выходных |
""")

st.subheader("Предобработка данных (EDA)")
st.markdown("""
**Шаги предобработки, выполненные в EDA:**

1. **Удаление служебного столбца** `Unnamed: 0`.
2. **Обработка пропусков**:
   - Числовые признаки заполнены **медианой** по столбцу.
   - Пропуски в столбце `date` заполнены методом **forward fill**, затем пропущенные временные метки скорректированы (+10 минут).
3. **Преобразование типов**:
   - Столбец `date` приведён к типу `datetime`.
   - Из `date` извлечены признаки: `Year`, `Month`, `Day_of_week`, `Hour`, `Minut`.
   - Созданы бинарные признаки:
     - `Weekend` – 1 для субботы и воскресенья.
     - `Night` (0–5 ч), `Morning` (6–11 ч), `Afternoon` (12–17 ч), `Evening` (18–23 ч).
   - Числовые признаки оптимизированы по памяти: `float64` → `float32`, `int64` → `int8/int16` где возможно.
   - Бинарные признаки приведены к типу `bool`.
4. **Проверка дубликатов** – дубликатов нет.
5. **Анализ выбросов** – например, признак `lights` имеет 100% выбросов по критерию IQR (много нулей), что соответствует реальному характеру использования освещения.
""")

st.subheader("Целевая переменная: Appliances")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Среднее", f"{df['Appliances'].mean():.1f} Вт·ч")
with col2:
    st.metric("Медиана", f"{df['Appliances'].median():.1f} Вт·ч")
with col3:
    st.metric("Максимум", f"{df['Appliances'].max():.0f} Вт·ч")
with col4:
    st.metric("Квартиль 75%", f"{df['Appliances'].quantile(0.75):.0f} Вт·ч")

st.subheader("Корреляции признаков с Appliances")
numeric_cols = df.select_dtypes(include=['number']).columns
correlations = df[numeric_cols].corr()['Appliances'].drop('Appliances').sort_values(ascending=False)
st.dataframe(correlations.to_frame("Корреляция"), use_container_width=True)

st.markdown("""
**Наиболее сильные корреляции (по модулю):**
- `T_out` (температура на улице) – положительная корреляция (чем теплее, тем выше энергопотребление, вероятно из‑за кондиционирования).
- `Tdewpoint` и `RH_out` – также заметно влияют.
- Внутренние температуры (`T1`–`T9`) показывают среднюю положительную связь.
- `lights` – слабая корреляция, что ожидаемо (освещение — не главный потребитель).
""")

st.subheader("Среднее энергопотребление по часам и дням недели")
pivot = df.pivot_table(values="Appliances", index="Hour", columns="Day_of_week", aggfunc="mean")
st.dataframe(pivot.round(1), use_container_width=True)
st.caption("Дни недели: 0 – понедельник, 6 – воскресенье. Виден пик в вечерние часы (17–20 ч) и более низкое потребление ночью.")

st.subheader("Основные выводы EDA")
st.markdown("""
- Распределение `Appliances` сильно скошено вправо: большинство наблюдений в диапазоне 10–100 Вт·ч, но есть редкие пики до 1000+ Вт·ч.
- Признаки `T_out`, `RH_out`, `Tdewpoint` имеют заметную корреляцию с целевой переменной.
- Выбросы в `lights` (много нулей) – норма для реальных данных; модель должна это учитывать.
- Временные признаки (`Hour`, `Day_of_week`, `Weekend`) помогают улавливать суточные и недельные циклы потребления.
- Многие признаки имеют пропуски (например, `RH_2` – 166 пропусков), которые были заполнены медианой.
""")

st.caption("EDA выполнен в Jupyter Notebook на основе данных `energy_task.csv`. Подготовлено для Streamlit-приложения.")
