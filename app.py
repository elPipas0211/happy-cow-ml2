# -*- coding: utf-8 -*-
"""
Happy Cow Ice Cream · Predicción de sabores top seller con MTC
Dashboard analítico — Machine Learning II
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Happy Cow · MTC Analytics",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(BASE, "modelo_happycow.pkl")
XLSX = os.path.join(BASE, "icecream1.xlsx")

# ── Paleta — 3 colores. Negro · blanco cálido · dorado ──────────────
INK      = "#0F0F0F"   # fondo — negro casi puro
INK_2    = "#161616"   # sidebar / paneles
INK_3    = "#1A1A1A"   # cards
LINEA    = "#2A2A2A"   # bordes
CREMA    = "#F5F0E8"   # texto principal — blanco cálido
CREMA_DK = "#A0978A"   # texto secundario — gris cálido
OCRE     = "#C8A96E"   # acento único — dorado suave
MUTE     = "#6E6258"   # gris profundo (solo escalas de gráfico)

# Escala monocroma para gráficos multi-serie (dorado → grises)
SEQ = [OCRE, CREMA_DK, MUTE, "#4A453F"]

NAT_LABEL  = {"Frio": "Frío", "Neutro": "Neutro", "Calido": "Cálido"}
NAT_COLOR  = {"Frío": CREMA_DK, "Neutro": CREMA_DK, "Cálido": CREMA_DK}
FUN_LABEL  = {"Generar_fluidos": "Generar fluidos", "Tonico": "Tónico",
              "Drenar_humedad": "Drenar humedad"}
MAL_LABEL  = {"Wind": "Viento", "Summer_Heat": "Calor de Verano",
              "Dampness": "Humedad", "Dryness": "Sequedad", "Cold": "Frío"}

ALINEACION_MTC = {
    "Wind": ["Calido", "Neutro"],
    "Summer_Heat": ["Frio"],
    "Dampness": ["Calido"],
    "Dryness": ["Frio", "Neutro"],
    "Cold": ["Calido"],
}

MTC_LOOKUP = {
    "Ginger": ("Calido", "Drenar_humedad", "Oriental"),
    "Red Bean": ("Calido", "Tonico", "Oriental"),
    "Green Tea": ("Frio", "Generar_fluidos", "Oriental"),
    "YY Sesame": ("Neutro", "Tonico", "Oriental"),
    "Chai Tea": ("Calido", "Drenar_humedad", "Oriental"),
    "Lime Coconut": ("Frio", "Generar_fluidos", "Oriental"),
    "Pure Coconut": ("Frio", "Generar_fluidos", "Oriental"),
    "Mango": ("Frio", "Generar_fluidos", "Oriental"),
    "Pina Colada": ("Frio", "Generar_fluidos", "Occidental"),
    "Mint Choco": ("Frio", "Generar_fluidos", "Occidental"),
    "Chocolate": ("Calido", "Tonico", "Occidental"),
    "Coffee": ("Calido", "Tonico", "Occidental"),
    "Hazelnut": ("Calido", "Tonico", "Occidental"),
    "Vanilla Bean": ("Neutro", "Tonico", "Occidental"),
    "Salted Caramel": ("Neutro", "Tonico", "Occidental"),
    "Strawberry": ("Frio", "Generar_fluidos", "Occidental"),
    "Banana Caramel": ("Neutro", "Tonico", "Occidental"),
    "Apricot": ("Neutro", "Generar_fluidos", "Occidental"),
    "Cherry Almond Fudge": ("Calido", "Tonico", "Occidental"),
    "Pistachio": ("Neutro", "Tonico", "Occidental"),
}

# ════════════════════════════════════════════════════════════════════
#  ESTILOS — monocromático: negro · blanco cálido · dorado
# ════════════════════════════════════════════════════════════════════
def inject_css():
    css = f"""
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700;900&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

:root {{
  --ink:{INK}; --ink2:{INK_2}; --ink3:{INK_3};
  --crema:{CREMA}; --cremadk:{CREMA_DK};
  --ocre:{OCRE}; --linea:{LINEA};
}}

/* ---- Fondo: negro plano con viñeta monocroma sutil ---- */
.stApp {{
  background:
    radial-gradient(1100px 620px at 82% -12%, {INK_2}, {INK} 62%);
  background-attachment: fixed;
}}
.stApp::before {{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  opacity:.20; mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/><feColorMatrix type='saturate' values='0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
}}

html, body, [class*="css"], .stApp, p, span, div, li, td, th, label {{
  font-family:'DM Sans', sans-serif; color:{CREMA};
}}
.block-container {{ padding:2.4rem 3.2rem 5rem; max-width:1500px; position:relative; z-index:1; }}
section.main .block-container {{ padding-top: 3.5rem !important; }}

h1,h2,h3 {{ font-family:'Playfair Display', serif; color:{CREMA}; letter-spacing:.2px; }}

/* ---- Sidebar ---- */
section[data-testid='stSidebar'] {{
    background: #161616;
    border-right: 1px solid #2A2A2A;
    min-width: 240px !important;
    max-width: 240px !important;
}}
section[data-testid='stSidebar'] .block-container {{ padding-top:1.6rem; }}
section[data-testid='stSidebar'] label {{
    display: flex;
    align-items: center;
    padding: 0.6rem 1rem;
    margin: 0.1rem 0;
    border-radius: 8px;
    border-left: 2px solid transparent;
    color: #A0978A;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s ease;
}}
section[data-testid='stSidebar'] label:hover {{
    color: #F5F0E8;
    background: rgba(200,169,110,0.08);
}}
section[data-testid='stSidebar'] label[data-checked='true'],
section[data-testid='stSidebar'] label:has(input:checked) {{
    border-left: 2px solid #C8A96E;
    color: #C8A96E;
    background: rgba(200,169,110,0.1);
    font-weight: 500;
}}
section[data-testid='stSidebar'] input {{ display: none; }}
section[data-testid='stSidebar'] div[role='radiogroup'] input[type='radio'] {{ display: none !important; }}

/* ---- Tarjetas / superficies ---- */
.hc-card {{
  background:{INK_3};
  border:1px solid {LINEA}; border-radius:14px; padding:1.4rem 1.6rem;
  box-shadow:0 14px 34px -28px rgba(0,0,0,.9);
}}
.hc-tag {{
  display:inline-block; font-family:'DM Sans'; font-weight:600;
  text-transform:uppercase; letter-spacing:.22em; font-size:.7rem;
  color:{OCRE}; border:1px solid {LINEA};
  padding:.28rem .8rem; border-radius:100px; background:transparent;
}}

/* ---- Tablas ---- */
.stDataFrame, .stTable {{ border-radius:12px; overflow:hidden; }}
table {{ font-size:.9rem !important; }}
thead tr th {{
  background:{INK_3} !important; color:{CREMA_DK} !important;
  font-family:'DM Sans' !important; text-transform:uppercase;
  letter-spacing:.1em; font-size:.74rem !important; font-weight:600 !important;
  border-bottom:1px solid {LINEA} !important;
}}
tbody tr td {{ background:{INK_2} !important; border-color:{LINEA} !important; }}
tbody tr:hover td {{ background:{INK_3} !important; }}

/* ---- Inputs ---- */
.stSelectbox div[data-baseweb="select"]>div, .stDateInput input,
.stMultiSelect div[data-baseweb="select"]>div {{
  background:{INK_3} !important; border-color:{LINEA} !important;
  border-radius:10px !important; color:{CREMA} !important;
}}
div[data-baseweb="popover"] li:hover {{ background:{INK_3} !important; }}

/* ---- Barras propias ---- */
.hc-bar-track {{ background:{LINEA}; border-radius:100px; height:8px; overflow:hidden; }}
.hc-bar-fill {{ height:100%; border-radius:100px; background:{OCRE}; }}

/* ---- Métricas ---- */
[data-testid="stMetricValue"] {{ font-family:'Playfair Display'; color:{OCRE}; }}

hr {{ border-color:{LINEA}; }}
::-webkit-scrollbar {{ width:9px; height:9px; }}
::-webkit-scrollbar-thumb {{ background:{LINEA}; border-radius:6px; }}
::selection {{ background:{OCRE}; color:{INK}; }}

@keyframes rise {{ from{{opacity:0; transform:translateY(14px);}} to{{opacity:1; transform:translateY(0);}} }}
.rise {{ animation:rise .6s cubic-bezier(.2,.7,.2,1) both; }}
.d1{{animation-delay:.05s}} .d2{{animation-delay:.12s}} .d3{{animation-delay:.19s}}
.d4{{animation-delay:.26s}} .d5{{animation-delay:.33s}}
"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    with open(PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_eda():
    """Reconstruye los insumos del EDA desde icecream1.xlsx (lógica del notebook)."""
    sabores_student = {
        'Apricot ST': 'Apricot', 'Banana Caramel St': 'Banana Caramel',
        'Chai Tea st': 'Chai Tea', 'Cherry Almond Fudge st': 'Cherry Almond Fudge',
        'Chocolate St': 'Chocolate', 'Coffee  St': 'Coffee', 'Ginger St': 'Ginger',
        'Green Tea St': 'Green Tea', 'Hazelnut St': 'Hazelnut',
        'Lime Coconut ST': 'Lime Coconut', 'Mango St': 'Mango',
        'Mint Choco  St': 'Mint Choco', 'Pina Colada St': 'Pina Colada',
        'Pistachio St': 'Pistachio', 'Pure Coco . St': 'Pure Coconut',
        'Red Bean St': 'Red Bean', 'S. Caramel St': 'Salted Caramel',
        'Strawberry St': 'Strawberry', 'Vanilla Bean St': 'Vanilla Bean',
        'YY Seasame St': 'YY Sesame',
    }
    sabores_staff = {
        'Apricot Staff': 'Apricot', 'Banana Staff': 'Banana Caramel',
        'Chai Tea Staff': 'Chai Tea', 'Cherry Alm Staff': 'Cherry Almond Fudge',
        'Chocolate Staff': 'Chocolate', 'Coffee Staff': 'Coffee',
        'Ginger Staff': 'Ginger', 'Green Tea Staff': 'Green Tea',
        'Hezelnut Staff': 'Hazelnut', 'Lime Coconut Staff': 'Lime Coconut',
        'Mango Staff': 'Mango', 'Mint Choco Staff': 'Mint Choco',
        'Pina Colada Staff': 'Pina Colada', 'Pistachio Staff': 'Pistachio',
        'Pure Coconut Staff': 'Pure Coconut', 'Red Bean Staff': 'Red Bean',
        'S. Caramel Staff': 'Salted Caramel', 'Strawberry Staff': 'Strawberry',
        'Vanilla Bean Staff': 'Vanilla Bean', 'YY Seasame Staff': 'YY Sesame',
    }
    df_student = pd.read_excel(XLSX, sheet_name='student daily')
    df_staff = pd.read_excel(XLSX, sheet_name='staff daily').rename(columns={'Product': 'Date'})

    st_df = df_student[['Date'] + list(sabores_student)].rename(columns=sabores_student)
    sf_df = df_staff[['Date'] + list(sabores_staff)].rename(columns=sabores_staff)
    sabores = list(sabores_student.values())
    st_df[sabores] = st_df[sabores].clip(lower=0)
    sf_df[sabores] = sf_df[sabores].clip(lower=0)

    def parse(serie):
        f = pd.to_datetime(serie, format='%a %dst %b %Y', errors='coerce')
        for fmt in ('%a %dnd %b %Y', '%a %drd %b %Y', '%a %dth %b %Y'):
            f = f.fillna(pd.to_datetime(serie, format=fmt, errors='coerce'))
        return f

    _st = st_df.copy(); _st['student_total'] = _st[sabores].sum(axis=1)
    _st['fecha'] = parse(_st['Date'])
    _sf = sf_df.copy(); _sf['staff_total'] = _sf[sabores].sum(axis=1)
    _sf['fecha'] = parse(_sf['Date'])
    daily = _st[['fecha', 'student_total']].merge(
        _sf[['fecha', 'staff_total']], on='fecha', how='inner')
    daily['local_total'] = daily['student_total'] + daily['staff_total']
    daily = daily.sort_values('fecha').reset_index(drop=True)
    return daily


def fig_layout(fig, h=420, title=None):
    fig.update_layout(
        template="plotly_dark", height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=CREMA, size=13),
        title=dict(text=title, font=dict(family="Playfair Display", size=19, color=CREMA)) if title else None,
        margin=dict(l=20, r=20, t=50 if title else 24, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
        colorway=SEQ, hoverlabel=dict(font_family="DM Sans", bgcolor=INK_3),
    )
    fig.update_xaxes(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA)
    fig.update_yaxes(gridcolor=LINEA, zerolinecolor=LINEA, linecolor=LINEA)
    return fig


def conclusion(text):
    st.markdown(f"""
<div style="border-left:2px solid {OCRE}; background:{INK_3};
     border-radius:0 12px 12px 0; padding:.85rem 1.2rem; margin:.6rem 0 1.8rem;">
  <span style="font-family:'DM Sans';text-transform:uppercase;letter-spacing:.18em;
        font-size:.7rem;color:{OCRE};font-weight:600;">Lectura del negocio</span><br>
  <span style="color:{CREMA_DK};font-size:.93rem;line-height:1.6;">{text}</span>
</div>""", unsafe_allow_html=True)


def section_head(tag, title, sub):
    st.markdown(f"""
<div class="rise" style="margin-bottom:1.6rem;">
  <span class="hc-tag">{tag}</span>
  <h1 style="font-size:2.5rem;margin:.5rem 0 .3rem;line-height:1.1;">{title}</h1>
  <p style="color:{CREMA_DK};font-size:1.02rem;max-width:760px;margin:0;">{sub}</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 1 · INTRODUCCIÓN
# ════════════════════════════════════════════════════════════════════
def tab_intro():
    st.markdown(f"""
<div class="rise" style="text-align:center;padding:2rem 0 1rem;">
  <span class="hc-tag">Machine Learning II · Proyecto Final</span>
  <h1 style="font-size:4.6rem;line-height:1.02;margin:1rem 0 .4rem;
       font-weight:900;letter-spacing:-1px;">
    Happy Cow <span style="color:{OCRE};font-style:italic;">Ice Cream</span>
  </h1>
  <p style="font-family:'Playfair Display';font-style:italic;font-size:1.5rem;
       color:{CREMA_DK};margin:0;">
    Predicción de sabores <span style="color:{OCRE};">top seller</span> en el segmento
    local de Hong Kong</p>
  <p style="color:{CREMA_DK};max-width:680px;margin:1rem auto 0;font-size:.95rem;">
    Un modelo de clasificación que cruza el calendario académico de HKU con la
    Medicina Tradicional China para decidir qué sabores merecen un lugar en la carta.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown(f"""
<div class="hc-card rise d1">
  <span class="hc-tag">El caso</span>
  <h3 style="margin:.7rem 0 .5rem;">Una tienda, un mercado por conquistar</h3>
  <p style="color:{CREMA_DK};line-height:1.7;">
   Mary Schroeder, CEO de <b style="color:{CREMA};">Happy Cow Ice Cream</b>, busca dejar
   de ser una marca percibida como producto para expatriados y penetrar el
   <b style="color:{CREMA};">mercado local chino</b> de Hong Kong. Su punto de prueba es la
   tienda de la <b style="color:{CREMA};">Universidad de Hong Kong (HKU)</b>, donde el
   <b style="color:{OCRE};">80%</b> de los clientes son consumidores locales —estudiantes
   y personal universitario.</p>
  <p style="color:{CREMA_DK};line-height:1.7;">
   El reto es operativo y concreto: el espacio de almacenamiento es reducido y el
   helado es un producto perecedero. Cada sabor mal elegido se traduce en
   <b style="color:{CREMA};">merma</b>, costos de obsolescencia y oportunidad comercial perdida.</p>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="hc-card rise d2">
  <span class="hc-tag">Tensión cultural</span>
  <h3 style="margin:.7rem 0 .5rem;">Oriente y Occidente en un mismo cono</h3>
  <p style="color:{CREMA_DK};line-height:1.7;">
   Hong Kong es una ciudad donde conviven la tradición china y la influencia
   occidental —y la universidad es su epicentro. El consumidor local no elige un
   helado solo por su sabor: lo elige según su <b style="color:{CREMA};">equilibrio
   térmico</b> y la temporada.</p>
  <p style="color:{CREMA_DK};line-height:1.7;">
   El proyecto convierte esa creencia cultural en una <b style="color:{CREMA};">variable
   medible</b> y la pone a competir con la analítica de datos moderna.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div class="hc-card rise">
  <span class="hc-tag">Marco conceptual · Medicina Tradicional China</span>
  <h3 style="margin:.7rem 0 .8rem;">El consumidor que piensa en términos de equilibrio</h3>
  <p style="color:{CREMA_DK};line-height:1.75;">
   Según la <b style="color:{CREMA};">Medicina Tradicional China (MTC)</b>, los alimentos no
   son neutros: cada uno actúa sobre el cuerpo. Las decisiones de compra del
   consumidor local de HKU están influenciadas por este marco, que clasifica los
   alimentos en tres dimensiones y los asocia con seis males estacionales descritos
   en los <b style="color:{OCRE};">Exhibits 1 y 2</b> del caso de Harvard Business School.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cc = st.columns(3)
    dims = [
        ("Naturaleza térmica", "Frío · Neutro · Cálido",
         "Cada alimento enfría o calienta el cuerpo. El consumidor busca contrarrestar el clima."),
        ("Función energética", "Generar fluidos · Drenar humedad · Tónico",
         "El efecto del alimento sobre la energía vital y los fluidos corporales."),
        ("Los seis males", "Viento · Calor de Verano · Humedad · Sequedad · Frío",
         "Condiciones climáticas estacionales que el alimento adecuado ayuda a equilibrar."),
    ]
    for col, (t, s, d) in zip(cc, dims):
        col.markdown(f"""
<div class="hc-card rise" style="height:215px;">
  <div style="width:36px;height:3px;background:{OCRE};border-radius:4px;margin-bottom:.8rem;"></div>
  <h3 style="font-size:1.2rem;margin:0 0 .3rem;">{t}</h3>
  <p style="color:{OCRE};font-weight:500;font-size:.86rem;margin:0 0 .6rem;">{s}</p>
  <p style="color:{CREMA_DK};font-size:.9rem;line-height:1.6;">{d}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    cp1, cp2 = st.columns([1, 1], vertical_alignment="top")
    with cp1:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">Problema de negocio</span>
  <h3 style="margin:.6rem 0 .5rem;">Decidir a ciegas cuesta caro</h3>
  <ul style="color:{CREMA_DK};line-height:1.8;padding-left:1.1rem;">
    <li><b style="color:{CREMA};">Espacio limitado</b> — solo caben pocos sabores en carta.</li>
    <li><b style="color:{CREMA};">Producto perecedero</b> — lo no vendido se pierde.</li>
    <li><b style="color:{CREMA};">Sin sistema predictivo</b> — hoy se decide por intuición,
        generando merma por sobrestock y oportunidad perdida.</li>
  </ul>
</div>""", unsafe_allow_html=True)
    with cp2:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">Pregunta analítica</span>
  <p style="font-family:'DM Sans';font-size:1.1rem;font-weight:400;
       line-height:1.65;color:{CREMA};margin:1rem 0;">
   “¿Qué características del sabor, del contexto académico y de la temporada
   climática ayudan a predecir si un sabor estará entre los más vendidos del
   segmento local?”</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size:1.9rem;'>Stakeholders del proyecto</h2>",
                unsafe_allow_html=True)
    stk = pd.DataFrame({
        "Stakeholder": ["Mary Schroeder (CEO)", "Equipo operativo HKU",
                        "Proveedores de ingredientes", "Consumidores locales (Student + Staff)"],
        "Rol en el proyecto": ["Tomadora de decisiones operativas", "Usuario directo del modelo",
                               "Afectados por la planificación de inventario",
                               "Fuente de los datos y beneficiarios indirectos"],
        "Interés principal": ["Definir la carta diaria sin generar merma",
                              "Recibir recomendaciones claras y accionables",
                              "Previsibilidad en los pedidos",
                              "Encontrar una oferta culturalmente relevante"],
    })
    st.dataframe(stk, hide_index=True, width='stretch')

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size:1.9rem;'>Flujo del proyecto · CRISP-DM</h2>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CREMA_DK};margin-bottom:1.4rem;'>"
                f"El proyecto sigue la metodología CRISP-DM, de la comprensión del "
                f"negocio al despliegue del modelo.</p>", unsafe_allow_html=True)
    steps = [
        ("01", "Negocio", "Caso Happy Cow, MTC y pregunta analítica"),
        ("02", "Datos", "icecream1.xlsx · Student + Staff · 183 días"),
        ("03", "Preparación", "Formato largo, features de serie de tiempo, MTC"),
        ("04", "Modelado", "Stacking LR + DT + RF, meta Regresión Logística"),
        ("05", "Evaluación", "Threshold óptimo, métricas y OKRs"),
        ("06", "Despliegue", "Este dashboard para el equipo de Mary"),
    ]
    cols = st.columns(6)
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f"""
<div class="hc-card rise" style="height:200px;padding:1.1rem;text-align:center;">
  <div style="font-family:'Playfair Display';font-size:2.4rem;font-weight:900;
       color:{OCRE};line-height:1;">{n}</div>
  <div style="width:24px;height:2px;background:{LINEA};margin:.5rem auto;border-radius:3px;"></div>
  <h3 style="font-size:1.05rem;margin:.3rem 0;">{t}</h3>
  <p style="color:{CREMA_DK};font-size:.8rem;line-height:1.5;">{d}</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 2 · DATOS
# ════════════════════════════════════════════════════════════════════
def tab_datos(data):
    df_ml = data["df_ml"]
    section_head("02 · Datos",
                 "De un Excel operativo a una base analítica",
                 "El dataset original mezcla sabores con formatos y toppings. "
                 "Esta sección documenta cómo se depuró hasta llegar a una base limpia "
                 "de 3.660 observaciones listas para modelar.")

    c = st.columns(4)
    metrics = [("183", "días observados", "Abr–Sep 2017"),
               ("20", "sabores válidos", "depurados del catálogo"),
               ("3.660", "filas finales", "183 días × 20 sabores"),
               ("2", "segmentos", "Student + Staff")]
    for col, (v, l, s) in zip(c, metrics):
        col.markdown(f"""
<div class="hc-card rise" style="text-align:center;height:140px;">
  <div style="font-family:'Playfair Display';font-size:2.6rem;font-weight:900;
       color:{OCRE};line-height:1;">{v}</div>
  <div style="font-weight:600;margin-top:.3rem;">{l}</div>
  <div style="color:{CREMA_DK};font-size:.8rem;">{s}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">El dataset</span>
  <h3 style="margin:.6rem 0;">icecream1.xlsx</h3>
  <p style="color:{CREMA_DK};line-height:1.7;">
   Registros diarios de ventas entre el <b style="color:{CREMA};">1 de abril</b> y el
   <b style="color:{CREMA};">30 de septiembre de 2017</b>, en dos hojas:
   <b style="color:{OCRE};">student daily</b> (183×33) y
   <b style="color:{OCRE};">staff daily</b> (183×30). Se corrige un error de
   nomenclatura: en <i>staff</i> la columna de fecha venía como <code>Product</code>.
   Las ventas locales se construyen sumando ambos segmentos.</p>
</div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">¿Por qué se excluyen turistas?</span>
  <h3 style="margin:.6rem 0;">El modelo solo mira al consumidor local</h3>
  <p style="color:{CREMA_DK};line-height:1.7;">
   El segmento <b style="color:{CREMA};">Tourism</b> se excluye del modelo principal: los
   turistas <b style="color:{CREMA};">no responden a la lógica de la MTC</b> ni siguen el
   <b style="color:{CREMA};">calendario académico de HKU</b>. Incluirlos introduciría ruido
   ajeno a la hipótesis cultural. El modelo se enfoca en <i>Student + Staff</i>,
   que representan el 80% local del caso.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size:1.7rem;'>Sabores válidos vs. complementos excluidos</h2>",
                unsafe_allow_html=True)
    cv1, cv2 = st.columns([1.4, 1])
    with cv1:
        sab = sorted(MTC_LOOKUP.keys())
        grid = "".join(
            f"<span style='display:inline-block;margin:.2rem;padding:.4rem .8rem;"
            f"background:{INK_2};border:1px solid {LINEA};color:{CREMA};"
            f"border-radius:8px;font-size:.85rem;'>{s}</span>" for s in sab)
        st.markdown(f"""
<div class="hc-card">
  <span class="hc-tag">20 sabores comercializables ✓</span>
  <div style="margin-top:.8rem;">{grid}</div>
</div>""", unsafe_allow_html=True)
    with cv2:
        exc = ["Single Scoop", "Double Scoop", "Triple Scoop", "Tub",
               "Toppings", "Vouchers", "Tarjetas / redenciones"]
        grid = "".join(
            f"<span style='display:inline-block;margin:.2rem;padding:.4rem .8rem;"
            f"background:transparent;border:1px solid {LINEA};color:{CREMA_DK};"
            f"border-radius:8px;font-size:.85rem;'>{s}</span>" for s in exc)
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag" style="color:{CREMA_DK};">Excluidos ✗</span>
  <div style="margin-top:.8rem;">{grid}</div>
  <p style="color:{CREMA_DK};font-size:.82rem;margin-top:.7rem;">
   No son sabores: distorsionarían el ranking y el cálculo del objetivo.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size:1.7rem;'>Lookup table MTC · perfil de cada sabor</h2>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CREMA_DK};'>Tabla experta construida en el "
                f"feature engineering: cada sabor se representa por sus atributos "
                f"conceptuales, no por su nombre.</p>", unsafe_allow_html=True)
    rows = []
    for s, (nat, fun, ori) in sorted(MTC_LOOKUP.items()):
        rows.append({"Sabor": s, "Naturaleza": NAT_LABEL[nat],
                     "Función": FUN_LABEL[fun], "Origen": ori})
    mtc_df = pd.DataFrame(rows)

    def color_nat(v):
        return f"color:{CREMA_DK};font-weight:600;"

    def color_ori(v):
        return f"color:{OCRE if v=='Oriental' else CREMA_DK};font-weight:600;"

    sty = (mtc_df.style
           .map(color_nat, subset=["Naturaleza"])
           .map(color_ori, subset=["Origen"]))
    st.dataframe(sty, hide_index=True, width='stretch', height=740)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div class="hc-card">
  <span class="hc-tag">Construcción de la base final</span>
  <h3 style="margin:.6rem 0;">De formato ancho a formato largo</h3>
  <p style="color:{CREMA_DK};line-height:1.7;">
   La tabla original tiene una columna por sabor. Se transforma a
   <b style="color:{CREMA};">formato largo</b> — una fila por
   <i>sabor × día</i>. Sobre esa base se construye la variable objetivo
   <b style="color:{OCRE};">top_seller</b>: vale <b>1</b> si el sabor está entre los
   <b style="color:{CREMA};">5 más vendidos</b> del segmento local ese día, y <b>0</b> en
   caso contrario. Así, el problema de negocio se convierte en una
   <b style="color:{CREMA};">clasificación binaria</b>.</p>
  <p style="color:{CREMA_DK};margin-top:.6rem;">
   Resultado: <b style="color:{OCRE};">{len(df_ml):,}</b> filas ·
   <b style="color:{OCRE};">{df_ml['sabor'].nunique()}</b> sabores ·
   <b style="color:{OCRE};">{df_ml['fecha'].nunique()}</b> días.</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 3 · EDA
# ════════════════════════════════════════════════════════════════════
def tab_eda(data, daily):
    df_ml = data["df_ml"].copy()
    section_head("03 · Análisis exploratorio",
                 "Qué cuentan los datos antes de modelar",
                 "Cinco visualizaciones interactivas sobre la dinámica comercial del "
                 "segmento local: del ritmo diario al desbalance de la variable objetivo.")

    # ── 1. Evolución diaria + media móvil 7d ──────────────────────
    st.markdown("### 13.1 · Evolución diaria de ventas locales")
    cf = st.columns([3, 1])
    with cf[1]:
        meses = {4: "Abril", 5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto", 9: "Septiembre"}
        sel = st.multiselect("Filtrar meses", list(meses.values()),
                             default=list(meses.values()), key="eda1")
    dly = daily.copy()
    dly["mes"] = dly["fecha"].dt.month
    inv = {v: k for k, v in meses.items()}
    dly = dly[dly["mes"].isin([inv[m] for m in sel])] if sel else daily.copy()
    dly = dly.sort_values("fecha")
    dly["mm7"] = dly["local_total"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_bar(x=dly["fecha"], y=dly["local_total"], name="Ventas diarias",
                marker_color=CREMA_DK, opacity=.25)
    fig.add_scatter(x=dly["fecha"], y=dly["mm7"], name="Media móvil 7 días",
                    line=dict(color=OCRE, width=3.5))
    if len(dly):
        prom = dly["local_total"].mean()
        fig.add_hline(y=prom, line_dash="dash", line_color=CREMA_DK,
                      annotation_text=f"Promedio {prom:,.0f}",
                      annotation_font_color=CREMA_DK)
    cf[0].plotly_chart(fig_layout(fig, 430), width='stretch')
    conclusion("La serie diaria es muy volátil, pero la media móvil de 7 días revela "
               "fases claras: un arranque fuerte en abril, una desaceleración hacia "
               "julio-agosto y una recuperación en septiembre. La demanda cambia de "
               "nivel a lo largo del semestre — no es solo ruido operativo.")

    # ── 2. Composición mensual Student vs Staff ───────────────────
    st.markdown("### 13.2 · Composición mensual · Student vs Staff")
    seg = daily.copy()
    seg["mes"] = seg["fecha"].dt.to_period("M").dt.to_timestamp()
    mseg = seg.groupby("mes", as_index=False)[["student_total", "staff_total"]].sum()
    cf = st.columns([3, 1])
    with cf[1]:
        modo = st.radio("Vista", ["Volumen", "Participación %"], key="eda2")
    mlab = mseg["mes"].dt.strftime("%b %Y")
    fig = go.Figure()
    if modo == "Volumen":
        fig.add_bar(x=mlab, y=mseg["student_total"], name="Student", marker_color=OCRE)
        fig.add_bar(x=mlab, y=mseg["staff_total"], name="Staff", marker_color=CREMA_DK)
    else:
        tot = mseg["student_total"] + mseg["staff_total"]
        fig.add_bar(x=mlab, y=mseg["student_total"] / tot * 100, name="Student", marker_color=OCRE)
        fig.add_bar(x=mlab, y=mseg["staff_total"] / tot * 100, name="Staff", marker_color=CREMA_DK)
    fig.update_layout(barmode="stack")
    cf[0].plotly_chart(fig_layout(fig, 400), width='stretch')
    conclusion("El componente estudiantil sostiene la demanda: ≈69% del total local "
               "frente a ≈31% del staff. Pero la mezcla no es rígida — en junio el "
               "staff llega a superar el 50%. La tienda opera sobre una base "
               "estudiantil, lo que justifica usar el calendario académico como variable.")

    # ── 3. Perfil MTC vs top_seller ───────────────────────────────
    st.markdown("### 13.3 · Perfil MTC vs. top_seller")
    cf = st.columns([3, 1])
    with cf[1]:
        dim = st.selectbox("Dimensión MTC",
                           ["Naturaleza", "Función", "Origen"], key="eda3")
    colmap = {"Naturaleza": ("naturaleza_sabor", NAT_LABEL),
              "Función": ("funcion_sabor", FUN_LABEL),
              "Origen": ("origen_sabor", {"Oriental": "Oriental", "Occidental": "Occidental"})}
    col, lab = colmap[dim]
    g = df_ml.groupby(col)["top_seller"].agg(["mean", "count"]).reset_index()
    g["label"] = g[col].map(lambda x: lab.get(x, x))
    g["pct"] = g["mean"] * 100
    g = g.sort_values("pct", ascending=False)
    fig = go.Figure()
    fig.add_bar(x=g["label"], y=g["pct"], marker_color=OCRE,
                text=[f"{v:.1f}%" for v in g["pct"]], textposition="outside",
                textfont=dict(color=CREMA))
    fig.add_hline(y=df_ml["top_seller"].mean() * 100, line_dash="dot",
                  line_color=CREMA_DK, annotation_text="Tasa base global",
                  annotation_font_color=CREMA_DK)
    fig.update_yaxes(title="% de observaciones top_seller", range=[0, max(g["pct"]) * 1.25])
    cf[0].plotly_chart(fig_layout(fig, 400), width='stretch')
    conclusion("El perfil del sabor sí discrimina. Los sabores de naturaleza Frío "
               "(28,7%), función Generar fluidos (27,2%) y origen Occidental (27,6%) "
               "aparecen con mayor frecuencia entre los top sellers. Es señal "
               "exploratoria de que la dimensión cultural MTC contiene información útil.")

    # ── 4. Heatmap mes × sabor ────────────────────────────────────
    st.markdown("### 13.4 · Heatmap · intensidad de ventas por mes y sabor")
    hm = df_ml.copy()
    hm["mes"] = hm["fecha"].dt.month
    mmap = {4: "Abr", 5: "May", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Sep"}
    hm["mlab"] = hm["mes"].map(mmap)
    cf = st.columns([3, 1])
    with cf[1]:
        agg = st.radio("Métrica", ["Promedio de ventas", "Tasa top_seller"], key="eda4")
    val = "ventas" if agg == "Promedio de ventas" else "top_seller"
    piv = hm.pivot_table(index="sabor", columns="mlab", values=val, aggfunc="mean")
    piv = piv.reindex(columns=["Abr", "May", "Jun", "Jul", "Ago", "Sep"])
    orden = piv.mean(axis=1).sort_values().index
    piv = piv.reindex(orden)
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index,
        colorscale=[[0, INK_2], [1, OCRE]],
        colorbar=dict(title=agg)))
    cf[0].plotly_chart(fig_layout(fig, 560), width='stretch')
    conclusion("La intensidad no se reparte por igual: abril y septiembre concentran "
               "los valores más altos, mientras julio-agosto se enfrían. Sabores como "
               "Salted Caramel, Mango, Chocolate y Mint Choco destacan de forma "
               "consistente — el efecto temporal interactúa con la identidad del sabor.")

    # ── 5. Distribución top_seller ────────────────────────────────
    st.markdown("### 13.5 · Distribución de la variable objetivo")
    st.markdown(f"<p style='color:{CREMA_DK};margin:-.3rem 0 .7rem;'>"
                f"Distribución de la clase objetivo en la base de datos de modelado.</p>",
                unsafe_allow_html=True)
    cf = st.columns([2, 1.4])
    vc = df_ml["top_seller"].value_counts().sort_index()
    fig = go.Figure(go.Pie(
        labels=["No top seller", "Top seller"], values=vc.values, hole=.62,
        marker=dict(colors=[INK_3, OCRE], line=dict(color=LINEA, width=2)),
        textfont=dict(color=CREMA, size=15)))
    fig.update_layout(annotations=[dict(
        text=f"<b>75 / 25</b>", font=dict(family="Playfair Display", size=30, color=CREMA),
        showarrow=False)])
    cf[0].plotly_chart(fig_layout(fig, 380), width='stretch')
    with cf[1]:
        st.markdown(f"""
<div class="hc-card" style="margin-top:1rem;">
  <span class="hc-tag">El desbalance 75 / 25</span>
  <p style="color:{CREMA_DK};line-height:1.7;margin-top:.7rem;">
   El <b style="color:{CREMA};">25%</b> positivo no es un hallazgo del mercado: es una
   <b style="color:{CREMA};">consecuencia mecánica</b> de la regla del objetivo. Cada día
   hay 20 sabores y exactamente 5 son top seller — 5/20 = 25%.</p>
  <p style="color:{CREMA_DK};line-height:1.7;">
   Es un desbalance <b style="color:{CREMA};">moderado, no extremo</b>: permite usar
   modelos supervisados sin técnicas agresivas de remuestreo, pero obliga a mirar
   <b style="color:{OCRE};">precisión, recall y F1</b> en lugar de solo accuracy.</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 4 · TRATAMIENTO DE DATOS
# ════════════════════════════════════════════════════════════════════
def tab_tratamiento(data):
    section_head("04 · Tratamiento de datos",
                 "Ingeniería de variables sin mirar el futuro",
                 "Limpieza, features de serie de tiempo con shift(1), variables "
                 "cíclicas y un split que respeta los días completos.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">Limpieza</span>
  <h3 style="margin:.6rem 0;">Antes de construir nada</h3>
  <ul style="color:{CREMA_DK};line-height:1.85;padding-left:1.1rem;">
   <li>Valores negativos (devoluciones / ajustes contables) →
       <b style="color:{CREMA};">0</b> vía <code>clip(lower=0)</code>.</li>
   <li>Exclusión de complementos: Single/Double/Triple Scoop, Tub, toppings,
       vouchers y redenciones.</li>
   <li>Homologación de nombres entre <i>Student</i> y <i>Staff</i>; corrección de la
       columna de fecha mal nombrada en <i>Staff</i>.</li>
  </ul>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">Anti data leakage</span>
  <h3 style="margin:.6rem 0;">La regla del shift(1)</h3>
  <p style="color:{CREMA_DK};line-height:1.7;">
   Todas las features históricas se calculan con <code>shift(1)</code>: el modelo
   <b style="color:{CREMA};">solo ve información anterior</b> al día que intenta predecir.
   Nunca observa las ventas del mismo día. Sin esta precaución, el modelo
   “haría trampa” y las métricas serían ilusorias.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:1.7rem;'>Features de serie de tiempo construidas</h2>",
                unsafe_allow_html=True)
    feats = pd.DataFrame({
        "Feature": ["ventas_lag_1", "top_lag_1",
                    "ventas_roll_mean_{3,7,14,28}", "ventas_roll_sum_{3,7,14,28}",
                    "top_rate_roll_{3,7,14,28}", "ventas_expanding_mean",
                    "top_expanding_rate"],
        "Qué captura": ["Ventas del día anterior",
                        "Si fue top seller el día anterior",
                        "Promedio móvil de ventas en 3/7/14/28 días",
                        "Suma móvil de ventas en 3/7/14/28 días",
                        "Tasa histórica de éxito como top seller",
                        "Promedio de ventas acumulado desde el inicio",
                        "Tasa de éxito acumulada desde el inicio"],
        "Anti-leakage": ["shift(1)", "shift(1)", "shift(1) + rolling",
                         "shift(1) + rolling", "shift(1) + rolling",
                         "shift(1) + expanding", "shift(1) + expanding"],
    })
    st.dataframe(feats, hide_index=True, width='stretch')
    st.markdown(f"<p style='color:{CREMA_DK};font-size:.9rem;'>Las ventanas de 3, 7, "
                f"14 y 28 días capturan el comportamiento reciente del sabor a distintos "
                f"horizontes. Las features <i>expanding</i> acumulan toda la historia previa.</p>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">Variables cíclicas</span>
  <h3 style="margin:.6rem 0;">El tiempo es un círculo</h3>
  <p style="color:{CREMA_DK};line-height:1.7;">
   El día 365 está al lado del día 1, pero numéricamente parecen lejísimos.
   Para que el modelo entienda esa continuidad, se codifican
   <b style="color:{OCRE};">seno y coseno</b> del día y de la semana del año:</p>
  <div style="font-family:monospace;background:{INK};border:1px solid {LINEA};
       border-radius:8px;padding:.7rem;color:{CREMA_DK};font-size:.82rem;margin-top:.5rem;">
   dia_sin = sin(2π · día/365)<br>dia_cos = cos(2π · día/365)<br>
   semana_sin = sin(2π · semana/52)<br>semana_cos = cos(2π · semana/52)</div>
</div>""", unsafe_allow_html=True)
    with cc2:
        ang = np.linspace(0, 2 * np.pi, 365)
        fig = go.Figure()
        fig.add_scatter(x=np.cos(ang), y=np.sin(ang), mode="lines",
                        line=dict(color=OCRE, width=3), name="Ciclo anual")
        for d, nm in [(0, "1 Ene"), (90, "1 Abr"), (180, "Jul"), (270, "Oct")]:
            a = 2 * np.pi * d / 365
            fig.add_scatter(x=[np.cos(a)], y=[np.sin(a)], mode="markers+text",
                            marker=dict(color=CREMA_DK, size=11), text=[nm],
                            textposition="top center", showlegend=False,
                            textfont=dict(color=CREMA_DK))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, scaleanchor="x")
        cc2.plotly_chart(fig_layout(fig, 300, "Codificación cíclica del año"),
                         width='stretch')

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:1.7rem;'>Split · StratifiedGroupKFold por fecha</h2>",
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="hc-card">
  <p style="color:{CREMA_DK};line-height:1.7;">
   El split agrupa por <b style="color:{OCRE};">día completo</b>. Todos los 20 sabores de
   una fecha caen juntos en train <i>o</i> en test — nunca partidos. Si un mismo día
   quedara dividido, el modelo vería el contexto de ese día durante el entrenamiento
   y lo “reconocería” en la prueba. <code>StratifiedGroupKFold</code> además mantiene
   la proporción 75/25 en ambos conjuntos.</p>
</div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    sp = st.columns([1, 1])
    with sp[0]:
        st.markdown(f"""
<div class="hc-card" style="text-align:center;">
  <span class="hc-tag">✓ Correcto</span>
  <p style="color:{CREMA_DK};margin:.8rem 0 .5rem;">El día entero viaja junto</p>
  <div style="display:flex;gap:4px;justify-content:center;flex-wrap:wrap;">
   {''.join(f'<div style="width:13px;height:13px;border-radius:3px;background:{OCRE if i<14 else CREMA_DK};"></div>' for i in range(20))}
  </div>
  <p style="color:{OCRE};font-size:.8rem;margin-top:.6rem;">Día A → train · Día B → test</p>
</div>""", unsafe_allow_html=True)
    with sp[1]:
        st.markdown(f"""
<div class="hc-card" style="text-align:center;">
  <span class="hc-tag" style="color:{CREMA_DK};">✗ Incorrecto</span>
  <p style="color:{CREMA_DK};margin:.8rem 0 .5rem;">Día partido = leakage</p>
  <div style="display:flex;gap:4px;justify-content:center;flex-wrap:wrap;">
   {''.join(f'<div style="width:13px;height:13px;border-radius:3px;background:{CREMA_DK if i%2 else MUTE};"></div>' for i in range(20))}
  </div>
  <p style="color:{CREMA_DK};font-size:.8rem;margin-top:.6rem;">Mismo día en train y test</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:1.7rem;'>Preprocesamiento dentro del Pipeline</h2>",
                unsafe_allow_html=True)
    pp = st.columns(2)
    pp[0].markdown(f"""
<div class="hc-card" style="height:100%;">
  <h3 style="color:{OCRE};margin:0 0 .4rem;font-size:1.2rem;">OneHotEncoder</h3>
  <p style="color:{CREMA_DK};line-height:1.65;font-size:.92rem;">
   Para las 7 variables categóricas: sabor, día de la semana, periodo académico,
   mal estacional, naturaleza, función y origen. <code>handle_unknown='ignore'</code>
   tolera categorías nuevas en producción.</p>
</div>""", unsafe_allow_html=True)
    pp[1].markdown(f"""
<div class="hc-card" style="height:100%;">
  <h3 style="color:{OCRE};margin:0 0 .4rem;font-size:1.2rem;">StandardScaler</h3>
  <p style="color:{CREMA_DK};line-height:1.65;font-size:.92rem;">
   Para las 23 variables numéricas: estandariza a media 0 y desviación 1.
   Crítico para que la Regresión Logística trate todas las escalas por igual.</p>
</div>""", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CREMA_DK};margin-top:1rem;'>Ambos transformadores "
                f"viven dentro de un <code>ColumnTransformer</code> y este, a su vez, "
                f"dentro del <code>Pipeline</code>. Así el escalado se ajusta "
                f"<b style='color:{CREMA}'>solo con train</b> y se aplica a test sin "
                f"filtrar información.</p>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 5 · MODELO Y PREDICCIÓN
# ════════════════════════════════════════════════════════════════════
def tab_modelo(data):
    thr = data["threshold"]
    ranking = data["ranking_diario"].copy()
    ranking["fecha"] = pd.to_datetime(ranking["fecha"])
    section_head("05 · Modelo y predicción",
                 "Stacking: tres miradas, una decisión",
                 "Una arquitectura de ensamble que combina un modelo lineal, un árbol "
                 "interpretable y un bosque, coronados por un meta-modelo.")

    # ── Diagrama de arquitectura ──────────────────────────────────
    st.markdown("<h2 style='font-size:1.7rem;'>Arquitectura del Stacking</h2>",
                unsafe_allow_html=True)
    bases = [("LR", "Regresión Logística", "Relaciones lineales"),
             ("DT", "Árbol de Decisión", "Interacciones no lineales · interpretable"),
             ("RF", "Random Forest", "Ensamble robusto de árboles")]
    cols = st.columns([1, 1, 1])
    for col, (ab, nm, d) in zip(cols, bases):
        col.markdown(f"""
<div class="hc-card rise" style="text-align:center;min-height:190px;margin-bottom:2rem;
     display:flex;flex-direction:column;">
  <div style="font-family:'Playfair Display';font-size:1.7rem;font-weight:900;color:{OCRE};">{ab}</div>
  <h3 style="font-size:1.05rem;margin:.2rem 0;">{nm}</h3>
  <p style="color:{CREMA_DK};font-size:.82rem;line-height:1.5;">{d}</p>
  <div style="color:{CREMA_DK};font-size:.7rem;margin-top:auto;padding-top:.7rem;
       font-family:'DM Sans';letter-spacing:.15em;">MODELO BASE</div>
</div>""", unsafe_allow_html=True)
    st.markdown(f"""
<div style="text-align:center;font-size:1.4rem;color:{CREMA_DK};
     padding-top:1rem;padding-bottom:1rem;">▼ ▼ ▼</div>
<div class="hc-card rise" style="text-align:center;max-width:560px;margin:0 auto;
     border:1px solid {OCRE};">
  <span class="hc-tag">Meta-modelo</span>
  <h3 style="margin:.5rem 0 .3rem;">Regresión Logística</h3>
  <p style="color:{CREMA_DK};font-size:.9rem;margin:0;">
   Aprende a combinar las predicciones <i>out-of-fold</i> de los tres modelos base
   (validación cruzada interna de 5 folds) para emitir la probabilidad final.</p>
</div>
<div style="text-align:center;font-size:1.4rem;color:{CREMA_DK};margin:.2rem 0;">▼</div>
<div style="text-align:center;font-family:'Playfair Display';font-style:italic;
     font-size:1.25rem;color:{CREMA};">probabilidad de ser top seller</div>
""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    # ── Selector de fecha + cards ─────────────────────────────────
    st.markdown("<h2 style='font-size:1.7rem;'>Recomendación del día</h2>",
                unsafe_allow_html=True)
    fechas = sorted(ranking["fecha"].dt.date.unique())
    cs = st.columns([1, 2])
    with cs[0]:
        fsel = st.date_input("Selecciona una fecha de evaluación",
                             value=fechas[0], min_value=fechas[0],
                             max_value=fechas[-1], key="modfecha")
    if fsel not in fechas:
        st.warning(f"El {fsel} no pertenece al conjunto de prueba. "
                   f"Fechas disponibles: {fechas[0]} – {fechas[-1]} "
                   f"({len(fechas)} días evaluados).")
        nearest = min(fechas, key=lambda d: abs((d - fsel).days))
        st.info(f"Mostrando la fecha más cercana disponible: **{nearest}**")
        fsel = nearest
    dia = ranking[ranking["fecha"].dt.date == fsel].sort_values(
        "prob_top_seller", ascending=False).head(5)
    ctx = dia.iloc[0]
    st.markdown(f"""
<div style="margin:.4rem 0 1rem;">
  <span class="hc-tag">{pd.Timestamp(fsel).strftime('%A %d de %B, %Y')}</span>
  <span style="margin-left:.6rem;color:{CREMA_DK};">Periodo académico:
   <b style="color:{CREMA};">{ctx['periodo_academico']}</b> · Mal estacional:
   <b style="color:{OCRE};">{MAL_LABEL.get(ctx['mal_estacional'],ctx['mal_estacional'])}</b></span>
</div>""", unsafe_allow_html=True)

    cards = st.columns(5)
    for i, (col, (_, r)) in enumerate(zip(cards, dia.iterrows())):
        prob = r["prob_top_seller"]
        acerto = int(r["top_seller"]) == 1
        nat = NAT_LABEL.get(r["naturaleza_sabor"], r["naturaleza_sabor"])
        fun = FUN_LABEL.get(r["funcion_sabor"], r["funcion_sabor"])
        ico = "✓" if acerto else "✗"
        icol = OCRE if acerto else CREMA_DK
        txt = "Fue top seller" if acerto else "No fue top seller"
        bw = max(4, prob * 100)
        col.markdown(f"""
<div class="hc-card rise d{i+1}" style="padding:1.1rem;min-height:300px;">
  <div style="display:flex;justify-content:space-between;align-items:start;">
   <span style="font-family:'DM Sans';color:{OCRE};font-weight:700;">#{i+1}</span>
   <span style="color:{icol};font-size:1.3rem;font-weight:800;">{ico}</span>
  </div>
  <h3 style="font-size:1.15rem;margin:.3rem 0 .6rem;line-height:1.2;">{r['sabor']}</h3>
  <div style="font-family:'Playfair Display';font-size:1.9rem;font-weight:900;
       color:{OCRE};line-height:1;">{prob*100:.1f}%</div>
  <div style="color:{CREMA_DK};font-size:.74rem;margin-bottom:.5rem;">prob. top seller</div>
  <div class="hc-bar-track"><div class="hc-bar-fill" style="width:{bw}%;"></div></div>
  <div style="margin-top:.7rem;font-size:.78rem;line-height:1.8;">
   <div>Naturaleza · <b style="color:{CREMA};">{nat}</b></div>
   <div>Función · <b style="color:{CREMA};">{fun}</b></div>
   <div>Origen · <b style="color:{CREMA};">{r['origen_sabor']}</b></div>
  </div>
  <div style="margin-top:.6rem;padding-top:.5rem;border-top:1px solid {LINEA};
       color:{icol};font-size:.78rem;font-weight:600;">{ico} {txt}</div>
</div>""", unsafe_allow_html=True)

    hits = int(dia["top_seller"].sum())
    st.markdown(f"<p style='text-align:center;color:{CREMA_DK};margin-top:1rem;'>"
                f"En este día el modelo acertó <b style='color:{OCRE};font-size:1.2rem;'>"
                f"{hits} de 5</b> recomendaciones.</p>", unsafe_allow_html=True)

    # ── Predicción libre · sabor nuevo ────────────────────────────
    st.markdown("<br><h2 style='font-size:1.7rem;'>Predice un sabor nuevo</h2>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CREMA_DK};margin-bottom:1rem;'>"
                f"Ingresa el perfil MTC de un sabor que no está en el histórico y una "
                f"fecha de evaluación. El modelo estima su probabilidad de ser top "
                f"seller a partir de ese perfil cultural y del contexto académico.</p>",
                unsafe_allow_html=True)

    nat_map = {"Frío": "Frio", "Neutro": "Neutro", "Cálido": "Calido"}
    fun_map = {"Drenar humedad": "Drenar_humedad",
               "Generar fluidos": "Generar_fluidos", "Tónico": "Tonico"}

    pc = st.columns(5)
    in_nat = pc[0].selectbox("Naturaleza", list(nat_map.keys()), key="pf_nat")
    in_fun = pc[1].selectbox("Función", list(fun_map.keys()), key="pf_fun")
    in_ori = pc[2].selectbox("Origen", ["Oriental", "Occidental"], key="pf_ori")
    in_fecha = pc[3].date_input("Fecha de evaluación",
                                value=pd.Timestamp("2017-05-15").date(), key="pf_fecha")
    in_per = pc[4].selectbox("Período académico",
                             ["Teaching", "Reading", "Assessment", "Break"], key="pf_per")

    if st.button("Predecir", key="pf_btn"):
        ts = pd.Timestamp(in_fecha)
        # Mal estacional estimado: moda del histórico para ese mes
        dfh = data["df_ml"].copy()
        dfh["fecha"] = pd.to_datetime(dfh["fecha"])
        mal_mes = (dfh.groupby(dfh["fecha"].dt.month)["mal_estacional"]
                   .agg(lambda s: s.mode().iloc[0]))
        mal = mal_mes.get(ts.month, dfh["mal_estacional"].mode().iloc[0])

        doy = int(ts.dayofyear)
        sem = int(ts.isocalendar().week)
        DOW = ["Monday", "Tuesday", "Wednesday", "Thursday",
               "Friday", "Saturday", "Sunday"]
        feat = {c: 0.0 for c in data["numeric_features"]}
        feat.update({
            "dia_del_año": doy, "mes": ts.month, "semana_del_año": sem,
            "dia_sin": np.sin(2 * np.pi * doy / 365),
            "dia_cos": np.cos(2 * np.pi * doy / 365),
            "semana_sin": np.sin(2 * np.pi * sem / 52),
            "semana_cos": np.cos(2 * np.pi * sem / 52),
        })
        feat.update({
            "sabor": "Nuevo",
            "dia_de_semana": DOW[ts.weekday()],
            "periodo_academico": in_per,
            "mal_estacional": mal,
            "naturaleza_sabor": nat_map[in_nat],
            "funcion_sabor": fun_map[in_fun],
            "origen_sabor": in_ori,
        })
        cols = list(data["numeric_features"]) + list(data["categorical_features"])
        X = pd.DataFrame([{c: feat[c] for c in cols}])
        try:
            prob = float(data["modelo"].predict_proba(X)[0, 1])
        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")
            prob = None
        if prob is not None:
            THR_LIBRE = 0.51
            ok = prob >= THR_LIBRE
            c = OCRE if ok else CREMA_DK
            msg = ("Este perfil tiene alta probabilidad de ser top seller" if ok
                   else "Este perfil no se recomienda para esta fecha")
            st.markdown(f"""
<div class="hc-card" style="text-align:center;border:1px solid {c};margin-top:1rem;">
  <span class="hc-tag">Probabilidad estimada</span>
  <div style="font-family:'Playfair Display';font-size:4rem;font-weight:900;
       color:{c};line-height:1.1;margin:.4rem 0;">{prob*100:.1f}%</div>
  <p style="color:{c};font-size:1.05rem;font-weight:500;margin:0;">{msg}</p>
  <p style="color:{CREMA_DK};font-size:.85rem;margin:.5rem 0 0;">
   Umbral de decisión: {THR_LIBRE:.2f} · Mal estacional estimado para
   {ts.strftime('%B')}: <b style="color:{CREMA};">{MAL_LABEL.get(mal, mal)}</b></p>
</div>""", unsafe_allow_html=True)

    # ── Threshold ─────────────────────────────────────────────────
    st.markdown("<br><h2 style='font-size:1.7rem;'>El umbral de decisión</h2>",
                unsafe_allow_html=True)
    tt1, tt2 = st.columns([1, 1.3])
    with tt1:
        st.markdown(f"""
<div class="hc-card" style="height:100%;text-align:center;">
  <span class="hc-tag">Threshold seleccionado</span>
  <div style="font-family:'Playfair Display';font-size:4.5rem;font-weight:900;
       color:{OCRE};line-height:1.1;margin:.5rem 0;">{thr:.2f}</div>
  <p style="color:{CREMA_DK};line-height:1.65;font-size:.9rem;">
   Elegido automáticamente como el umbral que <b style="color:{CREMA};">maximiza el
   F1-score</b> sobre el conjunto de prueba. El F1 equilibra precisión y recall:
   ni demasiada merma, ni demasiadas oportunidades perdidas.</p>
</div>""", unsafe_allow_html=True)
    with tt2:
        st.markdown(f"""
<div class="hc-card" style="height:100%;">
  <span class="hc-tag">Trade-off FP vs FN</span>
  <div style="display:flex;gap:1rem;margin-top:.8rem;">
   <div style="flex:1;border-left:2px solid {LINEA};padding-left:.8rem;">
    <b style="color:{CREMA};">Falso Positivo</b>
    <p style="color:{CREMA_DK};font-size:.85rem;line-height:1.55;margin:.3rem 0;">
     Se recomienda un sabor que no se vende → <b style="color:{CREMA};">merma</b> y
     costo operativo. Producto preparado que se pierde.</p>
   </div>
   <div style="flex:1;border-left:2px solid {LINEA};padding-left:.8rem;">
    <b style="color:{CREMA};">Falso Negativo</b>
    <p style="color:{CREMA_DK};font-size:.85rem;line-height:1.55;margin:.3rem 0;">
     No se recomienda un sabor que sí se habría vendido →
     <b style="color:{CREMA};">venta perdida</b>.</p>
   </div>
  </div>
  <p style="color:{CREMA_DK};font-size:.88rem;line-height:1.6;margin-top:.6rem;
       border-top:1px solid {LINEA};padding-top:.7rem;">
   En Happy Cow el <b style="color:{CREMA};">FP es más costoso</b>: el espacio de
   almacenamiento es limitado y un sabor que ocupa lugar sin venderse bloquea a otro
   que sí lo haría. Por eso interesa un umbral que no deje crecer los FP.</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 6 · MÉTRICAS
# ════════════════════════════════════════════════════════════════════
def tab_metricas(data):
    dtp = data["df_test_pred"].copy()
    thr = data["threshold"]
    y = dtp["top_seller"].values
    p = dtp["prob_top_seller"].values
    pred = (p >= thr).astype(int)
    section_head("06 · Métricas",
                 "Cómo de bueno es el modelo, sin maquillaje",
                 "Curvas, matriz de confusión e importancia de variables sobre las "
                 f"{dtp['fecha'].nunique()} fechas del conjunto de prueba.")

    # ── Curva de threshold ────────────────────────────────────────
    st.markdown("### Curva de threshold · Precisión · Recall · F1")
    ths = np.round(np.arange(0.0, 1.001, 0.01), 2)
    prec = [precision_score(y, (p >= t).astype(int), zero_division=0) for t in ths]
    rec = [recall_score(y, (p >= t).astype(int), zero_division=0) for t in ths]
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ths]
    fig = go.Figure()
    fig.add_scatter(x=ths, y=prec, name="Precisión", line=dict(color=CREMA_DK, width=2.5))
    fig.add_scatter(x=ths, y=rec, name="Recall", line=dict(color=MUTE, width=2.5))
    fig.add_scatter(x=ths, y=f1s, name="F1-score", line=dict(color=OCRE, width=3.5))
    fig.add_vline(x=thr, line_dash="dash", line_color=CREMA,
                  annotation_text=f"Threshold = {thr:.2f}", annotation_font_color=CREMA)
    fig.update_xaxes(title="Umbral de decisión"); fig.update_yaxes(title="Métrica", range=[0, 1.02])
    st.plotly_chart(fig_layout(fig, 420), width='stretch')
    conclusion(f"En el umbral {thr:.2f} se maximiza el F1: subir el umbral mejora la "
               "precisión pero sacrifica recall, y bajarlo hace lo contrario. Es el "
               "punto de equilibrio entre merma y oportunidades perdidas.")

    # ── Matriz de confusión ───────────────────────────────────────
    st.markdown("### Matriz de confusión")
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    mc1, mc2 = st.columns([1, 1])
    # Convención estándar: TP arriba-izq, FN arriba-der, FP abajo-izq, TN abajo-der.
    # En go.Heatmap la primera fila de z se dibuja abajo, por eso z[0] es la fila inferior.
    z = [[fp, tn], [tp, fn]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred. Top", "Pred. No top"], y=["Real No top", "Real Top"],
        text=[[f"FP<br>{fp}", f"TN<br>{tn}"], [f"TP<br>{tp}", f"FN<br>{fn}"]],
        texttemplate="%{text}", textfont=dict(size=18, color=CREMA),
        colorscale=[[0, INK_2], [1, OCRE]], showscale=False))
    mc1.plotly_chart(fig_layout(fig, 360), width='stretch')
    with mc2:
        cells = [
            ("TN", tn, "Verdadero Negativo", "Se descartó bien un sabor flojo."),
            ("FP", fp, "Falso Positivo", "Se recomendó un sabor que no vendió → merma."),
            ("FN", fn, "Falso Negativo", "Se descartó un sabor que sí vendía → venta perdida."),
            ("TP", tp, "Verdadero Positivo", "Se acertó: top seller correctamente recomendado."),
        ]
        for ab, v, nm, d in cells:
            mc2.markdown(f"""
<div style="border-left:2px solid {LINEA};padding:.4rem .9rem;margin:.4rem 0;
     background:{INK_3};border-radius:0 8px 8px 0;">
  <b style="color:{CREMA};">{ab} · {nm}</b>
  <span style="float:right;font-family:'Playfair Display';font-weight:900;color:{OCRE};">{v}</span>
  <p style="color:{CREMA_DK};font-size:.83rem;margin:.2rem 0 0;">{d}</p>
</div>""", unsafe_allow_html=True)

    # ── ROC + PR ──────────────────────────────────────────────────
    st.markdown("### Curvas ROC y Precision-Recall")
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    rc1, rc2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, name=f"ROC · AUC = {auc:.3f}",
                    line=dict(color=OCRE, width=3), fill="tozeroy",
                    fillcolor="rgba(200,169,110,.12)")
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Aleatorio",
                    line=dict(color=CREMA_DK, dash="dash"))
    fig.update_xaxes(title="FPR"); fig.update_yaxes(title="TPR")
    rc1.plotly_chart(fig_layout(fig, 380, f"ROC · AUC = {auc:.3f}"), width='stretch')
    fig = go.Figure()
    fig.add_scatter(x=rc, y=pr, name=f"PR · AP = {ap:.3f}",
                    line=dict(color=OCRE, width=3), fill="tozeroy",
                    fillcolor="rgba(200,169,110,.12)")
    fig.add_hline(y=y.mean(), line_dash="dash", line_color=CREMA_DK,
                  annotation_text="Tasa base", annotation_font_color=CREMA_DK)
    fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precisión")
    rc2.plotly_chart(fig_layout(fig, 380, f"Precision-Recall · AP = {ap:.3f}"),
                     width='stretch')
    conclusion(f"El AUC de {auc:.3f} indica que el modelo ordena bien: dado un par "
               "top/no-top al azar, casi siempre asigna mayor probabilidad al correcto. "
               f"El Average Precision de {ap:.3f} —más exigente con el desbalance 75/25— "
               "confirma capacidad real de discriminación, aunque con margen de mejora.")

    # ── Importancia de variables ──────────────────────────────────
    st.markdown("### Top 20 · Importancia de variables del árbol base")
    try:
        stk = data["modelo"].named_steps["model"]
        dt = stk.named_estimators_["dt"]
        prep = data["modelo"].named_steps["prep"]
        catn = list(prep.named_transformers_["cat"].get_feature_names_out(
            data["categorical_features"]))
        fnames = list(data["numeric_features"]) + catn
        imp = pd.Series(dt.feature_importances_, index=fnames)
        imp = imp[imp > 0].sort_values(ascending=False).head(20).sort_values()
        mtc_keys = ("naturaleza", "funcion", "origen", "mal_estacional")
        colors = [OCRE if any(k in n for k in mtc_keys) else CREMA_DK for n in imp.index]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker_color=colors))
        fig.update_xaxes(title="Importancia (Gini)")
        st.plotly_chart(fig_layout(fig, 520), width='stretch')
        st.markdown(f"<p style='color:{CREMA_DK};font-size:.88rem;'>"
                    f"<span style='color:{CREMA_DK};'>■</span> Serie de tiempo · "
                    f"<span style='color:{OCRE};'>■</span> Variables MTC</p>",
                    unsafe_allow_html=True)
        conclusion("La importancia la dominan las features de serie de tiempo —el "
                   "historial reciente de ventas del sabor (ventas_roll_mean_3, "
                   "top_expanding_rate, top_lag_1)—. Las variables MTC apenas aparecen: "
                   "el comportamiento reciente predice mejor que el perfil cultural, un "
                   "hallazgo clave que matiza la hipótesis del proyecto.")
    except Exception as e:
        st.info(f"No se pudo extraer la importancia de variables: {e}")

    # ── FP / FN por sabor ─────────────────────────────────────────
    st.markdown("### Falsos Positivos y Falsos Negativos por sabor")
    dtp["pred"] = pred
    dtp["fp"] = ((pred == 1) & (y == 0)).astype(int)
    dtp["fn"] = ((pred == 0) & (y == 1)).astype(int)
    err = dtp.groupby("sabor")[["fp", "fn"]].sum()
    err["total"] = err["fp"] + err["fn"]
    err = err.sort_values("total", ascending=False)
    fig = go.Figure()
    fig.add_bar(y=err.index, x=err["fp"], name="Falsos Positivos (merma)",
                orientation="h", marker_color=OCRE)
    fig.add_bar(y=err.index, x=err["fn"], name="Falsos Negativos (venta perdida)",
                orientation="h", marker_color=CREMA_DK)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Número de errores")
    st.plotly_chart(fig_layout(fig, 520), width='stretch')
    conclusion("Los sabores intermedios —los que rondan el límite del top 5— concentran "
               "la mayor confusión. Sabores claramente fuertes o claramente débiles casi "
               "no generan error; el modelo duda justo donde la frontera es difusa.")

    # ── Tabla de thresholds 0.3-0.7 ───────────────────────────────
    st.markdown("### Análisis de threshold · 0.30 a 0.70")
    rows = []
    for t in np.round(np.arange(0.30, 0.71, 0.05), 2):
        pr_ = (p >= t).astype(int)
        rows.append({"Threshold": f"{t:.2f}",
                     "Precisión": f"{precision_score(y,pr_,zero_division=0):.3f}",
                     "Recall": f"{recall_score(y,pr_,zero_division=0):.3f}",
                     "F1": f"{f1_score(y,pr_,zero_division=0):.3f}",
                     "Positivos predichos": int(pr_.sum())})
    tdf = pd.DataFrame(rows)

    def hl(row):
        is_sel = abs(float(row["Threshold"]) - thr) < 0.001
        return [f"background-color:rgba(200,169,110,.18);color:{OCRE};font-weight:700;"
                if is_sel else "" for _ in row]

    st.dataframe(tdf.style.apply(hl, axis=1), hide_index=True, width='stretch')
    st.markdown(f"<p style='color:{CREMA_DK};font-size:.88rem;'>"
                f"Fila resaltada: el threshold {thr:.2f} seleccionado por máximo F1.</p>",
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 7 · KPIs Y OKRs
# ════════════════════════════════════════════════════════════════════
def tab_kpis(data):
    dtp = data["df_test_pred"].copy()
    thr = data["threshold"]
    df_train_proxy = data["df_ml"]
    y = dtp["top_seller"].values
    p = dtp["prob_top_seller"].values
    pred = (p >= thr).astype(int)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    n_dias = dtp["fecha"].nunique()

    # KR3 — alineación MTC
    dtp["alineado"] = dtp.apply(
        lambda r: 1 if r["naturaleza_sabor"] in
        ALINEACION_MTC.get(r["mal_estacional"], []) else 0, axis=1)
    ts_real = dtp[dtp["top_seller"] == 1]
    kr3 = ts_real["alineado"].mean()
    mtc_por_mal = ts_real.groupby("mal_estacional")["alineado"].mean()

    # FP operacional + KR2
    top5_hist = set(df_train_proxy.groupby("sabor")["ventas"].mean().nlargest(5).index)
    fp_base = fp_mod = 0
    for f, dd in dtp.groupby("fecha"):
        reales = set(dd.sort_values("ventas", ascending=False).head(5)["sabor"])
        recom = set(dd.sort_values("prob_top_seller", ascending=False).head(5)["sabor"])
        fp_base += len(top5_hist - reales)
        fp_mod += len(recom - reales)
    kr2 = (fp_base - fp_mod) / fp_base if fp_base else 0

    section_head("07 · KPIs y OKRs",
                 "¿Cumplió el modelo lo que prometió?",
                 "Las metas se definieron antes de modelar. Aquí se confrontan con los "
                 "resultados reales sobre el conjunto de prueba.")

    # ── Metas definidas antes ─────────────────────────────────────
    st.markdown("<h2 style='font-size:1.6rem;'>Metas definidas antes del modelo</h2>",
                unsafe_allow_html=True)
    metas = pd.DataFrame({
        "Key Result": ["KR1 · Precisión operativa", "KR2 · Reducción de merma (FP)",
                        "KR3 · Validación hipótesis MTC"],
        "Meta": ["≥ 65%", "≥ 30%", "≥ 70%"],
        "Definición": [
            "≥65% de las recomendaciones del modelo están en el top 5 real",
            "Reducir 30% los días con FP frente a la línea base histórica",
            "≥70% de los top sellers reales con perfil MTC alineado a la temporada"],
    })
    st.dataframe(metas, hide_index=True, width='stretch')

    # ── KPIs reales ───────────────────────────────────────────────
    st.markdown("<br><h2 style='font-size:1.6rem;'>KPIs · valores reales</h2>",
                unsafe_allow_html=True)
    kpis = [
        ("Precisión del modelo", f"{precision*100:.1f}%",
         "De cada 10 recomendaciones, ~5-6 son correctas. Por debajo de la meta."),
        ("Tasa de merma diaria", f"{fp_mod} FP",
         f"{fp_mod} recomendaciones incorrectas en {n_dias} días ≈ "
         f"{fp_mod/n_dias:.1f} por día."),
        ("Alineación MTC", f"{kr3*100:.1f}%",
         f"Global {kr3*100:.1f}% · pero 71% bajo el mal estacional Wind."),
        ("Recall del modelo", f"{recall*100:.1f}%",
         "Detecta ~8 de cada 10 top sellers reales. Pocas oportunidades perdidas."),
    ]
    cols = st.columns(4)
    for i, (col, (t, v, d)) in enumerate(zip(cols, kpis)):
        col.markdown(f"""
<div class="hc-card rise d{i+1}" style="border-top:2px solid {OCRE};height:230px;">
  <span style="font-family:'DM Sans';text-transform:uppercase;
        letter-spacing:.12em;font-size:.74rem;color:{CREMA_DK};">{t}</span>
  <div style="font-family:'Playfair Display';font-size:3rem;font-weight:900;
       color:{OCRE};line-height:1.1;margin:.3rem 0;">{v}</div>
  <p style="color:{CREMA_DK};font-size:.84rem;line-height:1.55;">{d}</p>
</div>""", unsafe_allow_html=True)

    # ── OKRs con semáforo ─────────────────────────────────────────
    st.markdown("<br><h2 style='font-size:1.6rem;'>OKRs · cumplimiento</h2>",
                unsafe_allow_html=True)
    okrs = [
        ("KR1", "Precisión ≥ 65%", precision, 0.65, precision >= 0.65),
        ("KR2", "Reducción FP ≥ 30%", kr2, 0.30, kr2 >= 0.30),
        ("KR3", "Alineación MTC ≥ 70%", kr3, 0.70, kr3 >= 0.70),
    ]
    cols = st.columns(3)
    for col, (kr, nm, val, meta, ok) in zip(cols, okrs):
        c = OCRE if ok else CREMA_DK
        ico = "✓" if ok else "✗"
        estado = "Cumplido" if ok else "No cumplido"
        bw = min(100, val / meta * 100)
        col.markdown(f"""
<div class="hc-card rise">
  <div style="display:flex;justify-content:space-between;align-items:center;">
   <span class="hc-tag" style="color:{c};">{kr}</span>
   <span style="font-size:1.6rem;color:{c};font-weight:900;">{ico}</span>
  </div>
  <h3 style="margin:.6rem 0 .2rem;font-size:1.15rem;">{nm}</h3>
  <div style="font-family:'Playfair Display';font-size:2.6rem;font-weight:900;
       color:{c};line-height:1.1;">{val*100:.1f}%</div>
  <div class="hc-bar-track" style="margin:.4rem 0;">
   <div class="hc-bar-fill" style="width:{bw}%;background:{c};"></div></div>
  <div style="color:{CREMA_DK};font-size:.8rem;">Meta {meta*100:.0f}% ·
   <b style="color:{c};">{estado}</b></div>
</div>""", unsafe_allow_html=True)

    # ── Interpretación por KR ─────────────────────────────────────
    st.markdown("<br><h2 style='font-size:1.6rem;'>Interpretación de negocio</h2>",
                unsafe_allow_html=True)
    interp = [
        ("KR1", precision >= 0.65, f"Precisión del {precision*100:.1f}% — no cumplido",
         "La precisión queda bajo la meta del 65%. La causa es estructural: el "
         "dataset cubre solo 7 meses y 3 de los 6 males estacionales. El modelo no "
         "alcanza a aprender el patrón MTC completo. Con un ciclo anual completo se "
         "espera superar la meta."),
        ("KR2", kr2 >= 0.30, f"Reducción de FP del {kr2*100:.1f}% — cumplido",
         "Es el único KR cumplido. Frente a la selección manual histórica, el modelo "
         "reduce los falsos positivos por encima del 30%. Esto valida una mejora "
         "operativa real: menos producto preparado sin vender."),
        ("KR3", kr3 >= 0.70, f"Alineación MTC del {kr3*100:.1f}% — no cumplido globalmente",
         "Globalmente no llega al 70%, pero el desglose cuenta otra historia: bajo el "
         "mal estacional Wind la alineación alcanza el 71%. La hipótesis MTC es válida "
         "en primavera, pero en verano el calor domina la compra independientemente "
         "del perfil del sabor."),
    ]
    for kr, ok, titulo, txt in interp:
        c = OCRE if ok else CREMA_DK
        st.markdown(f"""
<div class="hc-card" style="border-left:2px solid {c};margin-bottom:.8rem;">
  <b style="color:{c};font-size:1.05rem;">{kr} · {titulo}</b>
  <p style="color:{CREMA_DK};line-height:1.7;margin:.4rem 0 0;">{txt}</p>
</div>""", unsafe_allow_html=True)

    # ── Desglose MTC por mal estacional ───────────────────────────
    st.markdown("<br><h2 style='font-size:1.6rem;'>Alineación MTC por mal estacional</h2>",
                unsafe_allow_html=True)
    md = mtc_por_mal.reset_index()
    md["label"] = md["mal_estacional"].map(lambda x: MAL_LABEL.get(x, x))
    md["pct"] = md["alineado"] * 100
    md = md.sort_values("pct", ascending=False)
    fig = go.Figure()
    fig.add_bar(x=md["label"], y=md["pct"],
                marker_color=[OCRE if v >= 70 else CREMA_DK for v in md["pct"]],
                text=[f"{v:.0f}%" for v in md["pct"]], textposition="outside",
                textfont=dict(color=CREMA))
    fig.add_hline(y=70, line_dash="dash", line_color=OCRE,
                  annotation_text="Meta KR3 · 70%", annotation_font_color=OCRE)
    fig.update_yaxes(title="% top sellers alineados con MTC", range=[0, 100])
    st.plotly_chart(fig_layout(fig, 380), width='stretch')
    conclusion("Solo Wind (Viento) supera la meta del 70%. Summer Heat y Dampness se "
               "quedan en ≈40%. El poder explicativo de la MTC depende de que la "
               "temporada tenga suficiente variación en los datos — algo que el "
               "horizonte de abril-septiembre no garantiza.")

    # ── Monitoreo ─────────────────────────────────────────────────
    st.markdown("<br><h2 style='font-size:1.6rem;'>Monitoreo y consideraciones éticas</h2>",
                unsafe_allow_html=True)
    mc = st.columns(3)
    mon = [
        ("Cuándo reentrenar",
         "Cuando lleguen datos de los males estacionales aún ausentes —"
         "<b>Cold</b> y <b>Dryness</b> (otoño-invierno)—. Solo con el ciclo anual "
         "completo el modelo puede aprender el patrón MTC íntegro."),
        ("Qué indicadores vigilar",
         "<b>Data drift</b> en las features de venta; cambios en la "
         "<b>distribución de ventas por sabor</b>; caída sostenida de precisión o "
         "recall respecto a esta línea base."),
        ("Consideraciones éticas",
         "El perfil MTC es una construcción experta, no un dato observado: no debe "
         "usarse para reforzar estereotipos culturales. El modelo apoya la decisión "
         "de Mary, no la sustituye."),
    ]
    for col, (t, d) in zip(mc, mon):
        col.markdown(f"""
<div class="hc-card" style="height:240px;border-top:2px solid {LINEA};">
  <h3 style="font-size:1.1rem;margin:.3rem 0 .4rem;color:{OCRE};">{t}</h3>
  <p style="color:{CREMA_DK};font-size:.87rem;line-height:1.6;">{d}</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
<div style="text-align:center;padding:.4rem 0 1rem;">
  <div style="font-family:'Playfair Display';font-weight:900;font-size:1.5rem;
       color:{CREMA};line-height:1.1;">Happy Cow</div>
  <div style="font-family:'Playfair Display';font-style:italic;color:{OCRE};
       font-size:.95rem;">Ice Cream · MTC</div>
  <div style="height:1px;background:{LINEA};margin:1rem 0 .3rem;"></div>
  <div style="font-family:'DM Sans';text-transform:uppercase;
       letter-spacing:.2em;font-size:.66rem;color:{CREMA_DK};">
   Predicción de top sellers</div>
</div>""", unsafe_allow_html=True)

        TABS = {
            "Introducción": tab_intro,
            "Datos": tab_datos,
            "EDA": tab_eda,
            "Tratamiento de datos": tab_tratamiento,
            "Modelo y Predicción": tab_modelo,
            "Métricas": tab_metricas,
            "KPIs y OKRs": tab_kpis,
        }
        choice = st.radio("Navegación", list(TABS.keys()),
                          label_visibility="collapsed")

    # Carga de datos con manejo de errores
    if not os.path.exists(PKL):
        st.error(f"⚠️ No se encontró **modelo_happycow.pkl** en:\n\n`{PKL}`\n\n"
                 "Coloca el archivo del modelo entrenado junto a `app.py` para "
                 "habilitar el dashboard.")
        st.stop()
    try:
        data = load_model()
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        st.stop()

    daily = None
    if choice == "EDA":
        if not os.path.exists(XLSX):
            st.error(f"⚠️ No se encontró **icecream1.xlsx** en `{XLSX}`. "
                     "Es necesario para las visualizaciones del EDA.")
            st.stop()
        try:
            daily = load_eda()
        except Exception as e:
            st.error(f"⚠️ Error al procesar icecream1.xlsx: {e}")
            st.stop()

    fn = TABS[choice]
    if choice == "Introducción":
        fn()
    elif choice == "EDA":
        fn(data, daily)
    else:
        fn(data)


if __name__ == "__main__":
    main()
