# Clasificador de Objetivos de Desarrollo Sostenible (ODS)

Aplicación web que clasifica automáticamente textos según los **17 Objetivos de Desarrollo Sostenible (ODS)** de la ONU, usando técnicas de Procesamiento de Lenguaje Natural y Machine Learning.

## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://importprogrammer-ods-classifier.streamlit.app/)

## Pipeline del modelo

```
Texto → Preprocesamiento → TF-IDF → TruncatedSVD (LSA) → Regresión Logística → ODS
```

| Componente                   | Detalle                                         |
| ---------------------------- | ----------------------------------------------- |
| Vectorización                | TF-IDF, bigramas, 20.000 features, sublinear_tf |
| Reducción de dimensionalidad | TruncatedSVD — 300 componentes (LSA)            |
| Clasificador                 | Logistic Regression multinomial                 |
| Optimización                 | GridSearchCV, 5-fold CV, métrica F1 Macro       |

## Resultados

| Métrica     | Valor |
| ----------- | ----- |
| Accuracy    | 88.0% |
| F1 Macro    | 85.2% |
| F1 Weighted | 87.9% |

Entrenado con el dataset [OSDG Community Dataset 2023](https://osdg.ai) — 9.656 textos en español etiquetados con los ODS 1–16.

## Estructura del repositorio

```
├── app.py              # Aplicación Streamlit
├── preprocessing.py    # Función de preprocesamiento de texto
├── model.joblib        # Modelo entrenado (serializado)
└── requirements.txt    # Dependencias
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```
