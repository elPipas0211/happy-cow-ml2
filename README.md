# Happy Cow Ice Cream · Dashboard MTC

Dashboard analítico del proyecto final de **Machine Learning II** — predicción de
sabores *top seller* en el segmento local de Hong Kong, cruzando el calendario
académico de HKU con la Medicina Tradicional China (MTC).

## Contenido

| Archivo | Descripción |
|---|---|
| `app.py` | Dashboard Streamlit (7 secciones: introducción, datos, EDA, tratamiento, modelo, métricas, KPIs). |
| `modelo_happycow.pkl` | Modelo de stacking entrenado + datos de evaluación. |
| `icecream1.xlsx` | Datos fuente de ventas (Student + Staff). |
| `ML2ProyectoFinalAriasNietoFinal.ipynb` | Notebook con el análisis completo. |
| `requirements.txt` | Dependencias con versiones fijadas. |

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Entra en [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta.
3. **New app** → selecciona el repositorio, rama `main` y archivo `app.py`.
4. En *Advanced settings*, elige **Python 3.13**.
5. Deploy. Streamlit instalará `requirements.txt` automáticamente.

> Las versiones de `scikit-learn` y `pandas` están fijadas para que el modelo
> serializado (`modelo_happycow.pkl`) se cargue sin errores de compatibilidad.

## Autores

Arias · Nieto — HKU Case Study, Machine Learning II.
