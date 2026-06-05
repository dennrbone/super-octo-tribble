import streamlit as st
from PIL import Image

st.set_page_config(page_title="РГР Россинский", layout="wide")
st.title("Web-приложение для инференса моделей ML")

col1, col2 = st.columns([1, 3])
with col1:
    try:
        photo = Image.open("assets/photo.jpg")
        st.image(photo, width=200)
    except:
        st.image("https://via.placeholder.com/200", caption="Фото не найдено")
with col2:
    st.markdown("**Разработчик:** Россинский Андрей Евгеньевич")
    st.markdown("**Группа:** ФИТ-241")
    st.markdown("**Тема:** Прогнозирование энергопотребления (регрессия)")

st.markdown("---")
st.markdown("Цель: разработать дашборд для предсказания энергопотребления (Appliances) на основе показаний датчиков.")