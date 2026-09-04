import streamlit as st
import numpy as np
from Analisis_Sismico import Analisis_Modal, Analisis_Espectral, Analisis_Bidireccional, m, cm, mm, kgf, tnf

st.set_page_config(page_title="Análisis Sísmico Dinámico Modal Espectral", layout="wide")

# ============================================================
# SIDEBAR: datos de entrada
# ============================================================
with st.sidebar:
    st.image("Image.jpg", width=150)
    st.header('Curso: Python aplicado a la Ingeniería Estructural')
    st.subheader('Creado por: Christian Velasco\n')
    st.caption('Análisis Dinámico Modal Espectral Bidireccional - Norma E.030 (2026)')
    st.write('')

    r1, r2 = st.columns([1, 2], gap="medium")
    with r1:
        "N° de niveles"
    with r2:
        nlevels = st.number_input("nlevels", value=4, min_value=1, step=1, label_visibility="collapsed")

    st.markdown("**Peso, rigidez y altura por entrepiso**")
    w = []
    kx = []
    ky = []
    h = []
    for i in range(int(nlevels)):
        st.markdown(f"*Nivel {i + 1}*")
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            wi = st.number_input(f"w{i+1}", value=120.0, min_value=0.01, step=1.0,
                                  key=f"w_{i+1}")
        with c2:
            kxi = st.number_input(f"kx{i+1}", value=25000.0, min_value=0.01, step=100.0,
                                   key=f"kx_{i+1}")
        with c3:
            kyi = st.number_input(f"ky{i+1}", value=22000.0, min_value=0.01, step=100.0,
                                   key=f"ky_{i+1}")
        with c4:
            hi = st.number_input(f"h{i+1}", value=3.0, min_value=0.1, step=0.1,
                                  key=f"h_{i+1}")
        cc1, cc2, cc3, cc4 = st.columns(4, gap="small")
        with cc1:
            st.caption("Peso (tnf)")
        with cc2:
            st.caption("Kx (tnf/m)")
        with cc3:
            st.caption("Ky (tnf/m)")
        with cc4:
            st.caption("h (m)")
            
        w.append(wi+1)
        kx.append(kxi+1)
        ky.append(kyi+1)
        h.append(hi+1)

    st.markdown("---")
    st.markdown("**Parámetros sísmicos (Norma E.030)**")

    g = st.number_input("Aceleración de la gravedad g (m/s²)", value=9.81, min_value=0.1, step=0.01)

    c1, c2 = st.columns(2)
    with c1:
        Z = st.number_input("Factor de zonificación Z", value=0.45, min_value=0.0, step=0.01)
        S = st.number_input("Factor de suelo S", value=1.10, min_value=0.0, step=0.01)
        Tp = st.number_input("Periodo Tp (s)", value=0.6, min_value=0.01, step=0.01)
    with c2:
        U = st.number_input("Factor de uso U", value=1.00, min_value=0.0, step=0.01)
        R = st.number_input("Coeficiente de reducción R", value=8.00, min_value=0.1, step=0.1)
        Tl = st.number_input("Periodo Tl (s)", value=2.0, min_value=0.01, step=0.01)

    run = st.button("Analizar", type="primary", use_container_width=True)

# ============================================================
# CUERPO PRINCIPAL
# ============================================================
st.title('Análisis Dinámico Modal Espectral Bidireccional')
st.caption('Adaptado del notebook de análisis sísmico según la Norma E.030 (2026)')

if not run:
    st.info("Completa los datos en la barra lateral y presiona **Analizar** para ejecutar el modelo.")
else:
    # Asignación de unidades MKS y conversión de pesos a masa
    m_masa = [wi * (tnf / g) for wi in w]
    kx_u = [kxi * (tnf / m) for kxi in kx]
    ky_u = [kyi * (tnf / m) for kyi in ky]
    h_u = [hi * m for hi in h]

    S_coeff = Z * U * S * g / R

    # ---------------- Análisis modal (informativo, eje X) ----------------
    x1 = Analisis_Modal(m_masa, kx_u, h_u)

    st.subheader("Resultados gráficos de los modos de vibración (eje X)")
    fig_modal = x1.Graficos()
    st.pyplot(fig_modal)

    st.subheader("Resultados numéricos de los modos de vibración")
    st.dataframe(x1.modes, use_container_width=True)

    st.subheader("Periodos y masas participativas")
    st.dataframe(x1.results, use_container_width=True)

    # ---------------- Análisis espectral bidireccional ----------------
    x2 = Analisis_Espectral(m_masa, kx_u, h_u, S_coeff, R, Tp, Tl)  # 100% sismo en X
    x3 = Analisis_Espectral(m_masa, ky_u, h_u, S_coeff, R, Tp, Tl)  # 30% sismo en Y

    xbi = Analisis_Bidireccional(x2, x3)

    st.subheader("Desplazamientos, derivas y cortantes bidireccionales")
    fig_bi = xbi.Graficos_bidireccionales()
    st.pyplot(fig_bi)

    st.subheader("Tabla de desplazamientos y derivas E.030 - 2026")
    st.dataframe(xbi.Tabla_Resultados(), use_container_width=True)
