import streamlit as st
import pandas as pd
import joblib

# Charger modèle et liste de features
model = joblib.load("model_xgb_arrest.pkl")
features_class = joblib.load("features_class.pkl")

st.title("Prédiction d'Arrestation - Crimes à Chicago")

st.sidebar.header("Entrer les caractéristiques du crime")

hour = st.sidebar.number_input("Heure (0-23)", min_value=0, max_value=23, value=22)
dayofweek = st.sidebar.number_input("Jour de la semaine (0=lundi ... 6=dimanche)", min_value=0, max_value=6, value=5)
month = st.sidebar.number_input("Mois (1-12)", min_value=1, max_value=12, value=12)
lat = st.sidebar.number_input("Latitude", value=41.88)
lon = st.sidebar.number_input("Longitude", value=-87.63)
primary_label = st.sidebar.number_input("PrimaryType_Label", value=15)

if st.button("Prédire"):
    # Construire observation
    new_obs = pd.DataFrame({
        "Hour": [hour],
        "DayOfWeek": [dayofweek],
        "Month": [month],
        "Latitude": [lat],
        "Longitude": [lon],
        "PrimaryType_Label": [primary_label]
    })

    # Ajouter les colonnes District_... à 0 si nécessaire
    for col in features_class:
        if col not in new_obs.columns:
            new_obs[col] = 0

    new_obs = new_obs[features_class]

    proba = model.predict_proba(new_obs)[0][1]
    pred = int(proba >= 0.5)

    st.subheader("Résultat")
    st.write(f"Probabilité d'arrestation : **{proba*100:.1f}%**")
    if pred == 1:
        st.success("Prédiction : Arrestation probable")
    else:
        st.info("Prédiction : Pas d'arrestation")
