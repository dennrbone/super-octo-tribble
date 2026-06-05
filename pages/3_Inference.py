import streamlit as st
import pandas as pd
import pickle
import joblib
import json
import umap
from catboost import CatBoostRegressor

@st.cache_resource
def load_all():
    # Основной scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Scaler для UMAP (повторная стандартизация после UMAP)
    try:
        with open("models/scaler_umap.pkl", "rb") as f:
            scaler_umap = pickle.load(f)
    except FileNotFoundError:
        scaler_umap = None
        st.warning("Файл scaler_umap.pkl не найден. ML6 может работать некорректно.")

    # Загрузка тренировочных данных для повторного обучения UMAP
    with open("models/X_train_scaled.pkl", "rb") as f:
        X_train_scaled = pickle.load(f)

    # Загрузка параметров UMAP из JSON
    with open("models/umap_params.json", "r") as f:
        umap_params = json.load(f)

    # Создание и обучение UMAP
    umap_transformer = umap.UMAP(**umap_params, random_state=42, n_jobs=-1)
    umap_transformer.fit(X_train_scaled)

    # Загрузка ML-моделей
    with open("models/ml1_linear.pkl", "rb") as f:
        ml1 = pickle.load(f)
    with open("models/ml2_gb.pkl", "rb") as f:
        ml2 = pickle.load(f)
    with open("models/ml4_bagging.pkl", "rb") as f:
        ml4 = pickle.load(f)
    with open("models/ml5_stacking.pkl", "rb") as f:
        ml5 = pickle.load(f)
    with open("models/ml6_mlp.pkl", "rb") as f:
        ml6 = joblib.load(f)

    ml3 = CatBoostRegressor()
    ml3.load_model("models/ml3_catboost.cbm")

    return scaler, scaler_umap, ml1, ml2, ml3, ml4, ml5, ml6, umap_transformer

scaler, scaler_umap, ml1, ml2, ml3, ml4, ml5, ml6, umap_transformer = load_all()

models_dict = {
    "Linear Regression (ML1)": (ml1, False),
    "Gradient Boosting (ML2)": (ml2, False),
    "CatBoost (ML3)": (ml3, False),
    "Bagging (ML4)": (ml4, False),
    "Stacking (ML5)": (ml5, False),
    "MLP (ML6)": (ml6, True)   # для ML6 применяем UMAP + второй scaler
}

st.set_page_config(page_title="Инференс", layout="wide")
st.title("Прогноз энергопотребления")

df_template = pd.read_csv("energyoutput.csv")
if "Unnamed: 0" in df_template.columns:
    df_template.drop("Unnamed: 0", axis=1, inplace=True)
for col in ['date', 'UTC', 'Data', 'timestamp']:
    if col in df_template.columns:
        df_template.drop(col, axis=1, inplace=True)

feature_names = [c for c in df_template.columns if c != "Appliances"]
boolean_cols = ['Weekend', 'Night', 'Morning', 'Afternoon', 'Evening']

model_name = st.selectbox("Выберите модель", list(models_dict.keys()))
model, use_umap = models_dict[model_name]
method = st.radio("Способ ввода", ["Ручной ввод", "Загрузить CSV"])

X_input = None

if method == "Ручной ввод":
    st.subheader("Введите значения всех признаков")
    cols_per_row = 3
    input_data = {}
    for i, col in enumerate(feature_names):
        col_idx = i % cols_per_row
        if col_idx == 0:
            cols = st.columns(cols_per_row)
        with cols[col_idx]:
            if col in boolean_cols:
                input_data[col] = 1 if st.checkbox(col, value=False) else 0
            else:
                dtype = df_template[col].dtype
                default_val = float(df_template[col].mean()) if dtype == 'float64' else int(df_template[col].mean())
                if dtype == 'float64':
                    input_data[col] = st.number_input(col, value=default_val, format="%.4f", key=col)
                else:
                    input_data[col] = st.number_input(col, value=default_val, key=col)
    if st.button("Спрогнозировать", type="primary"):
        X_input = pd.DataFrame([input_data])[feature_names]

else:  # Загрузка CSV
    uploaded = st.file_uploader("Загрузите CSV-файл с признаками", type="csv")
    if uploaded:
        X_input = pd.read_csv(uploaded)
        extra = [c for c in X_input.columns if c not in feature_names]
        if extra:
            st.warning(f"Удалены лишние столбцы: {extra}")
            X_input = X_input.drop(columns=extra)
        missing = [c for c in feature_names if c not in X_input.columns]
        if missing:
            st.error(f"Отсутствуют столбцы: {missing}")
            X_input = None
        else:
            X_input = X_input[feature_names]

if X_input is not None:
    try:
        # Шаг 1: стандартизация исходных признаков
        X_scaled = scaler.transform(X_input)
        
        # Шаг 2: для ML6 применяем UMAP + повторную стандартизацию
        if use_umap:
            X_umap = umap_transformer.transform(X_scaled)
            if scaler_umap is not None:
                X_umap = scaler_umap.transform(X_umap)
            else:
                st.error("Отсутствует scaler_umap.pkl, невозможно обработать данные для ML6")
                st.stop()
            X_final = X_umap
        else:
            X_final = X_scaled
        
        pred = model.predict(X_final)[0]
        st.success(f"**Прогноз энергопотребления:** {pred:.2f} Вт·ч")
    except Exception as e:
        st.error(f"Ошибка при прогнозировании: {e}")
        st.write("Колонки в X_input:", list(X_input.columns))
        st.write("Ожидаемые колонки (первые 10):", feature_names[:10])