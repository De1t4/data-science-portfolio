import streamlit as st
import joblib
import pandas as pd

# 1. Cargar el modelo (Descongelar el cerebro)
model = joblib.load('titanic_model.pkl')

# 2. Título y Descripción
st.title("🚢 Titanic Survival Predictor")
st.write("¿Habrías sobrevivido al Titanic? Averígualo con IA.")

# 3. Inputs del Usuario (La UI)
# Streamlit crea los sliders y botones automáticamente
pclass = st.selectbox("Clase del Pasajero", [1, 2, 3])
sex = st.selectbox("Sexo", ["Hombre", "Mujer"])
age = st.slider("Edad", 0, 100, 30)
fare = st.number_input("Precio del Ticket ($)", 0, 500, 32)
family_size = st.slider("Tamaño de Familia (Acompañantes + Tú)", 1, 10, 1)

# 4. Procesar los datos (Igual que en tu limpieza del Notebook)
# Importante: El input debe tener LA MISMA estructura que usaste para entrenar
sex_numeric = 0 if sex == "Hombre" else 1

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_numeric],
    'Age': [age],
    'Fare': [fare],
    'FamilySize': [family_size]
})

# 5. Botón de Predicción
if st.button("Calcular Destino"):
    # Predecir clase (0 o 1)
    prediction = model.predict(input_data)[0]
    # Predecir probabilidad (ej: 0.85)
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 ¡Sobreviviste! (Probabilidad: {probability:.1%})")
        st.balloons()
    else:
        st.error(f"💀 No sobreviviste... (Probabilidad: {probability:.1%})")


# Pie de página
st.markdown("---")
st.markdown("""
**Desarrollado con:**
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

**Fuente de datos:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
""")
