"""
app.py — DentalVision EDA Dashboard
Streamlit multi-page app con análisis exploratorio del dataset dental.
Deploy: Hugging Face Spaces (Docker, puerto 7860)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

from utils.data_loader import (
    CLASS_COLORS,
    CLASS_NAMES,
    compute_stats,
    get_sample_images,
    load_dataframes,
)

# ─── Config global ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DentalVision EDA",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ─────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #1e1e2e; }
  [data-testid="stSidebar"]          { background: #181825; }
  h1, h2, h3, h4                     { color: #cdd6f4; }
  .metric-card {
    background: #313244; border-radius: 12px;
    padding: 1rem 1.5rem; text-align: center;
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #89b4fa; }
  .metric-card .label { font-size: 0.85rem; color: #a6adc8; margin-top: 4px; }
  div[data-testid="stMetricValue"]   { color: #89b4fa; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#181825",
    font_color="#cdd6f4",
    xaxis=dict(gridcolor="#313244"),
    yaxis=dict(gridcolor="#313244"),
)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🦷 DentalVision EDA")
    st.markdown("---")
    page = st.selectbox(
        "Navegar a",
        [
            "🏠 Inicio",
            "📊 Distribución de Clases",
            "🔬 Propiedades de Imagen",
            "🔍 Calidad de Datos",
            "🖼️ Galería de Imágenes",
        ],
    )
    st.markdown("---")
    st.caption("Dataset: Dental X-Ray Classification")
    st.caption("Clases: Prótesis · Sano · Caries · Otro")


# ─── Carga de datos (cacheada) ─────────────────────────────────────────────
df_train, df_test = load_dataframes()

with st.spinner("Calculando estadísticas… (primera vez puede tardar ~3 min)"):
    df_train_stats, df_test_stats = compute_stats(df_train, df_test)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — INICIO
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.title("🦷 DentalVision — Dashboard EDA")
    st.markdown("Análisis exploratorio interactivo del dataset de imágenes dentales.")

    # Métricas globales
    cols = st.columns(5)
    metrics = [
        ("Total Train", f"{len(df_train):,}",  "#89b4fa"),
        ("Total Test",  f"{len(df_test):,}",   "#a6e3a1"),
        ("Clases",      "4",                   "#fab387"),
        ("Brillo Medio",f"{df_train_stats['brightness'].mean():.1f}", "#f38ba8"),
        ("Tamaño Medio",f"{df_train_stats['bytes_size'].mean()/1024:.0f} KB", "#cba6f7"),
    ]
    for col, (label, value, color) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="value" style="color:{color}">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c1, c2 = st.columns(2)

    # Distribución Train
    with c1:
        counts = df_train["label"].value_counts().sort_index()
        fig = px.pie(
            names=[CLASS_NAMES[i] for i in counts.index],
            values=counts.values,
            color_discrete_sequence=CLASS_COLORS,
            title="Clases en Train",
            hole=0.45,
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=True)
        fig.update_traces(textinfo="percent+label", textfont_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)

    # Tabla resumen
    with c2:
        st.markdown("#### Resumen del Dataset")
        summary = pd.DataFrame({
            "Split":          ["Train", "Test"],
            "Imágenes":       [f"{len(df_train):,}", f"{len(df_test):,}"],
            "% del total":    [
                f"{100*len(df_train)/(len(df_train)+len(df_test)):.1f}%",
                f"{100*len(df_test) /(len(df_train)+len(df_test)):.1f}%",
            ],
            "Clases únicas":  [
                df_train["label"].nunique(),
                df_test["label"].nunique() if "label" in df_test.columns else "—",
            ],
            "Nulos":          [
                df_train.isnull().sum().sum(),
                df_test.isnull().sum().sum(),
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("#### Distribución por Clase (Train)")
        dist = df_train["label"].value_counts().sort_index().reset_index()
        dist.columns = ["Clase", "N"]
        dist["Clase"] = dist["Clase"].map(lambda x: f"{x} — {CLASS_NAMES[x]}")
        dist["%"] = (dist["N"] / dist["N"].sum() * 100).round(1)
        st.dataframe(dist, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — DISTRIBUCIÓN DE CLASES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Distribución de Clases":
    st.title("📊 Distribución de Clases")

    # Filtros
    col_f1, col_f2 = st.columns(2)
    split = col_f1.radio("Split", ["Train", "Test", "Ambos"], horizontal=True)

    df_sel = {"Train": df_train, "Test": df_test, "Ambos": pd.concat([df_train, df_test])}[split]

    tab1, tab2, tab3 = st.tabs(["Frecuencias", "Comparativa Train vs Test", "Test estadístico"])

    with tab1:
        counts = df_sel["label"].value_counts().sort_index()
        fig = px.bar(
            x=[CLASS_NAMES[i] for i in counts.index],
            y=counts.values,
            color=[CLASS_NAMES[i] for i in counts.index],
            color_discrete_sequence=CLASS_COLORS,
            labels={"x": "Clase", "y": "Frecuencia"},
            title=f"Distribución de clases — {split}",
            text_auto=True,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        train_pct = [
            df_train["label"].value_counts(normalize=True).sort_index().get(i, 0) * 100
            for i in range(4)
        ]
        test_pct = [
            df_test["label"].value_counts(normalize=True).sort_index().get(i, 0) * 100
            for i in range(4)
        ] if "label" in df_test.columns else [0] * 4

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train", x=CLASS_NAMES, y=train_pct,
                             marker_color="#89b4fa", text=[f"{v:.1f}%" for v in train_pct],
                             textposition="outside"))
        fig.add_trace(go.Bar(name="Test",  x=CLASS_NAMES, y=test_pct,
                             marker_color="#fab387", text=[f"{v:.1f}%" for v in test_pct],
                             textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                          title="Distribución relativa (%) Train vs Test",
                          yaxis_title="%", xaxis_title="Clase")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        observed = df_train["label"].value_counts().sort_index().values
        chi2_val, p_val = stats.chisquare(observed)
        c1, c2, c3 = st.columns(3)
        c1.metric("χ² (chi-cuadrado)", f"{chi2_val:.2f}")
        c2.metric("p-valor", f"{p_val:.4f}")
        c3.metric("Desbalance", "⚠️ Sí" if p_val < 0.05 else "✅ No")

        if p_val < 0.05:
            st.warning("Las clases **NO** están balanceadas (p < 0.05). Se recomienda aplicar técnicas de balanceo (pesos, oversampling, etc.) al entrenar modelos.")
        else:
            st.success("Las clases están suficientemente balanceadas (p ≥ 0.05).")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — PROPIEDADES DE IMAGEN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Propiedades de Imagen":
    st.title("🔬 Propiedades de Imagen")

    # Filtro de clases
    selected_classes = st.multiselect(
        "Filtrar clases",
        options=list(range(4)),
        default=list(range(4)),
        format_func=lambda x: CLASS_NAMES[x],
    )
    df_s = df_train_stats[df_train_stats["label"].isin(selected_classes)]

    tab1, tab2, tab3, tab4 = st.tabs(["Brillo & Contraste", "Dimensiones", "Canales RGB", "Correlaciones"])

    with tab1:
        c1, c2 = st.columns(2)
        for ax_col, prop, title in [(c1, "brightness", "Brillo"), (c2, "contrast", "Contraste")]:
            fig = go.Figure()
            for cls in selected_classes:
                vals = df_s[df_s["label"] == cls][prop].values
                if len(vals) > 1:
                    kde = stats.gaussian_kde(vals)
                    xr  = np.linspace(vals.min(), vals.max(), 300)
                    fig.add_trace(go.Scatter(
                        x=xr, y=kde(xr),
                        name=CLASS_NAMES[cls],
                        line=dict(color=CLASS_COLORS[cls], width=2.5),
                        fill="tozeroy",
                        fillcolor=CLASS_COLORS[cls].replace("#", "rgba(") + ",0.07)" if not CLASS_COLORS[cls].startswith("rgba") else CLASS_COLORS[cls],
                    ))
            fig.update_layout(**PLOTLY_LAYOUT, title=f"KDE — {title}",
                              xaxis_title=title, yaxis_title="Densidad")
            ax_col.plotly_chart(fig, use_container_width=True)

        # Boxplots
        fig = go.Figure()
        for cls in selected_classes:
            fig.add_trace(go.Box(
                y=df_s[df_s["label"] == cls]["brightness"].values,
                name=CLASS_NAMES[cls],
                marker_color=CLASS_COLORS[cls],
                boxmean=True,
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Boxplot — Brillo por Clase")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        for col, (xp, yp) in [(c1, ("width","height")), (c2, ("aspect","bytes_size"))]:
            fig = px.scatter(
                df_s, x=xp, y=yp,
                color=df_s["label"].map(lambda x: CLASS_NAMES[x]),
                color_discrete_sequence=CLASS_COLORS,
                opacity=0.5, title=f"{xp} vs {yp}",
                labels={"color": "Clase"},
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            col.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Canal R", "Canal G", "Canal B"])
        ch_colors = ["#f38ba8", "#a6e3a1", "#89b4fa"]
        for ci, (ch, ch_color) in enumerate(zip(["mean_r","mean_g","mean_b"], ch_colors), 1):
            for cls in selected_classes:
                vals = df_s[df_s["label"] == cls][ch].values
                if len(vals) > 1:
                    kde = stats.gaussian_kde(vals)
                    xr  = np.linspace(vals.min(), vals.max(), 200)
                    fig.add_trace(go.Scatter(
                        x=xr, y=kde(xr),
                        name=CLASS_NAMES[cls],
                        line=dict(color=CLASS_COLORS[cls], width=2),
                        showlegend=(ci == 1),
                    ), row=1, col=ci)
        fig.update_layout(**PLOTLY_LAYOUT, title="Distribución RGB por Clase", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap medias RGB
        rgb_means = df_train_stats.groupby("label")[["mean_r","mean_g","mean_b"]].mean().round(1)
        rgb_means.index = CLASS_NAMES
        fig_hm = px.imshow(
            rgb_means.T,
            text_auto=True, color_continuous_scale="magma",
            labels={"x": "Clase", "y": "Canal"},
            title="Media RGB por Clase",
        )
        fig_hm.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab4:
        corr_cols = [c for c in
                     ["width","height","aspect","bytes_size","brightness","contrast",
                      "mean_r","mean_g","mean_b","std_r","std_g","std_b","label"]
                     if c in df_train_stats.columns]
        corr = df_train_stats[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Matriz de Correlación",
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        if "label" in corr.columns:
            st.markdown("#### Correlación con la etiqueta")
            top = corr["label"].drop("label").abs().sort_values(ascending=False).reset_index()
            top.columns = ["Propiedad", "|r| con Label"]
            top["|r| con Label"] = top["|r| con Label"].round(3)
            st.dataframe(top, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — CALIDAD DE DATOS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Calidad de Datos":
    st.title("🔍 Calidad de Datos")

    # Duplicados
    dup_train = df_train_stats[df_train_stats.duplicated(subset="md5", keep=False)]
    dup_test  = df_test_stats [df_test_stats .duplicated(subset="md5", keep=False)]
    leaked    = set(df_train_stats["md5"]) & set(df_test_stats["md5"])

    # Outliers IQR
    outlier_report = {}
    for col in ["brightness","contrast","bytes_size","width","height"]:
        if col not in df_train_stats.columns:
            continue
        Q1, Q3 = df_train_stats[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        n = len(df_train_stats[
            (df_train_stats[col] < Q1 - 1.5*IQR) |
            (df_train_stats[col] > Q3 + 1.5*IQR)
        ])
        outlier_report[col] = n

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicados Train", f"{len(dup_train)}",
              f"{len(dup_train)/len(df_train_stats)*100:.1f}%")
    c2.metric("Duplicados Test",  f"{len(dup_test)}",
              f"{len(dup_test)/len(df_test_stats)*100:.1f}%")
    c3.metric("Data Leakage", f"{len(leaked)} imgs",
              "⚠️ Riesgo" if leaked else "✅ Sin riesgo")
    c4.metric("Outliers (brillo)", outlier_report.get("brightness", 0))

    if leaked:
        st.error(f"⚠️ **{len(leaked)} imágenes** son idénticas (mismo MD5) en Train y Test — esto puede inflar métricas de evaluación.")
    else:
        st.success("✅ No hay data leakage entre Train y Test.")

    st.markdown("---")
    tab1, tab2 = st.tabs(["Duplicados", "Outliers"])

    with tab1:
        c1, c2 = st.columns(2)
        # % duplicados por split
        fig = px.bar(
            x=["Train", "Test"],
            y=[len(dup_train)/len(df_train_stats)*100, len(dup_test)/len(df_test_stats)*100],
            color=["Train", "Test"],
            color_discrete_sequence=["#89b4fa","#fab387"],
            labels={"x": "Split", "y": "% Duplicados"},
            title="% Imágenes Duplicadas (MD5)",
            text_auto=".1f",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        c1.plotly_chart(fig, use_container_width=True)

        # Duplicados por clase
        if len(dup_train) > 0:
            dup_cls = dup_train.groupby("label").size().reset_index(name="n")
            dup_cls["Clase"] = dup_cls["label"].map(lambda x: CLASS_NAMES[x])
            fig2 = px.bar(
                dup_cls, x="Clase", y="n",
                color="Clase",
                color_discrete_sequence=CLASS_COLORS,
                title="Duplicados por Clase (Train)",
                text_auto=True,
            )
            fig2.update_layout(**PLOTLY_LAYOUT)
            c2.plotly_chart(fig2, use_container_width=True)

    with tab2:
        out_df = pd.DataFrame(
            [(k, v, round(v/len(df_train_stats)*100, 1)) for k, v in outlier_report.items()],
            columns=["Propiedad", "N Outliers", "% del total"],
        )
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        for cls in range(4):
            fig.add_trace(go.Box(
                y=df_train_stats[df_train_stats["label"] == cls]["brightness"].values,
                name=CLASS_NAMES[cls],
                marker_color=CLASS_COLORS[cls],
                boxpoints="outliers",
                jitter=0.3,
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Outliers de Brillo por Clase (Train)")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — GALERÍA DE IMÁGENES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Galería de Imágenes":
    st.title("🖼️ Galería de Imágenes por Clase")

    col_f1, col_f2, col_f3 = st.columns(3)
    cls_sel  = col_f1.selectbox("Clase", range(4), format_func=lambda x: CLASS_NAMES[x])
    n_imgs   = col_f2.slider("N° de imágenes", 3, 12, 6)
    seed_val = col_f3.number_input("Semilla aleatoria", value=42, step=1)

    st.markdown(f"### {CLASS_NAMES[cls_sel]} — {n_imgs} muestras")

    with st.spinner("Cargando imágenes…"):
        images = get_sample_images(df_train, cls_sel, n=n_imgs, seed=int(seed_val))

    if not images:
        st.warning("No se pudieron cargar imágenes.")
    else:
        cols = st.columns(min(n_imgs, 4))
        for i, img in enumerate(images):
            col = cols[i % len(cols)]
            col.image(img, use_container_width=True,
                      caption=f"Muestra {i+1} · {img.width}×{img.height}")

    st.markdown("---")
    st.markdown("#### Histograma de Píxeles (primera imagen cargada)")
    if images:
        arr = np.array(images[0])
        fig = go.Figure()
        for ch_idx, (ch_name, ch_col) in enumerate(
            zip(["R","G","B"], ["#f38ba8","#a6e3a1","#89b4fa"])
        ):
            fig.add_trace(go.Histogram(
                x=arr[:,:,ch_idx].flatten(),
                nbinsx=64,
                name=ch_name,
                marker_color=ch_col,
                opacity=0.65,
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="overlay",
            title="Distribución de Píxeles RGB",
            xaxis_title="Intensidad (0-255)",
            yaxis_title="Frecuencia",
        )
        st.plotly_chart(fig, use_container_width=True)
