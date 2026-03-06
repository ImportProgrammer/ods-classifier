"""
Aplicación Streamlit — Clasificador de ODS
==========================================
Carga el modelo entrenado y permite clasificar texto libre.
"""

import numpy as np
import streamlit as st
import joblib
from preprocessing import preprocess, TextPreprocessor  # noqa: F401 — necesario para deserializar el modelo

ODS_NAMES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

ODS_COLORS = {
    1: "#E5243B", 2: "#DDA63A", 3: "#4C9F38", 4: "#C5192D",
    5: "#FF3A21", 6: "#26BDE2", 7: "#FCC30B", 8: "#A21942",
    9: "#FD6925", 10: "#DD1367", 11: "#FD9D24", 12: "#BF8B2E",
    13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B", 16: "#00689D",
    17: "#19486A",
}


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    return joblib.load("model.joblib")


def classify_text(model, text: str) -> dict:
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    top_idx = np.argsort(proba)[::-1][:5]
    return {
        "ods_predicho": int(classes[top_idx[0]]),
        "probabilidad": float(proba[top_idx[0]]),
        "top5": [(int(classes[i]), float(proba[i])) for i in top_idx],
    }


def main():
    st.set_page_config(
        page_title="Clasificador ODS",
        page_icon="🌍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # ── Encabezado ────────────────────────────────────────────────────────────
    st.title("🌍 Clasificador de Objetivos de Desarrollo Sostenible")
    st.markdown(
        """
        Esta aplicación utiliza un modelo de **Machine Learning** entrenado con técnicas
        de Procesamiento de Lenguaje Natural (NLP) para clasificar automáticamente un texto
        según los **17 Objetivos de Desarrollo Sostenible (ODS)** de la ONU.

        **Pipeline del modelo:** TF-IDF → LSA (TruncatedSVD) → Regresión Logística
        """
    )
    st.divider()

    # ── Entrada de texto ──────────────────────────────────────────────────────
    text_input = st.text_area(
        "Ingresa el texto a clasificar:",
        height=200,
        placeholder=(
            "Ejemplo: 'El acceso universal al agua potable y al saneamiento básico "
            "es fundamental para erradicar la pobreza y mejorar la salud pública...'"
        ),
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        classify_btn = st.button("Clasificar", type="primary", use_container_width=True)
    with col2:
        if st.button("Limpiar", use_container_width=True):
            st.rerun()

    # ── Resultado ─────────────────────────────────────────────────────────────
    if classify_btn:
        if not text_input.strip():
            st.warning("Por favor ingresa un texto antes de clasificar.")
        else:
            model = load_model()
            result = classify_text(model, text_input)

            ods_num = result["ods_predicho"]
            ods_name = ODS_NAMES.get(ods_num, "Desconocido")
            color = ODS_COLORS.get(ods_num, "#333333")
            prob = result["probabilidad"]

            st.divider()

            # Resultado principal
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 1.5rem 2rem;
                    border-radius: 12px;
                    text-align: center;
                    margin-bottom: 1rem;
                ">
                    <h2 style="margin:0; font-size: 1.2rem; opacity: 0.9;">ODS Predicho</h2>
                    <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">ODS {ods_num}</h1>
                    <h3 style="margin:0; font-size: 1.1rem;">{ods_name}</h3>
                    <p style="margin-top: 0.8rem; opacity: 0.85; font-size: 0.9rem;">
                        Confianza: {prob:.1%}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top 5 probabilidades
            st.subheader("Distribución de probabilidades (Top 5)")
            for ods_i, prob_i in result["top5"]:
                name_i = ODS_NAMES.get(ods_i, "")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(
                        prob_i,
                        text=f"ODS {ods_i} — {name_i}",
                    )
                with col_b:
                    st.markdown(
                        f"<div style='text-align:right; padding-top:6px;'>"
                        f"<b>{prob_i:.1%}</b></div>",
                        unsafe_allow_html=True,
                    )

    # ── Pie de página ─────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Modelo entrenado con el dataset OSDG Community Dataset (2023) | "
        "Pipeline: TF-IDF + LSA + Regresión Logística"
    )


if __name__ == "__main__":
    main()
