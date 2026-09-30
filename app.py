import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import json
from shapely.geometry import shape

st.set_page_config(page_title="Nursery - Exploratory Demo", layout="wide")

ORTHO_SCALE = 4.228

STD_MINIMA_AFFIDABILE = 0.03
NDVI_HIGH_THRESHOLD = 0.929
NDVI_LOW_THRESHOLD = 0.752
SOGLIA_STRESS = -1.5
SOGLIA_OTTIMALE = 1.0
CWSI_STRESS_THRESHOLD = 0.5

STATUS_COLORS = {'vigorous': '#2ecc71', 'normal': '#f1c40f', 'stressed': '#e74c3c'}


@st.cache_data
def load_data():
    inventory = pd.read_csv("inventario_app.csv")
    block_mapping = pd.read_csv("mappatura_blocchi.csv")

    with open("contorni_tutte_piante.geojson", encoding="utf-8-sig") as f:
        geojson_data = json.load(f)
    contours = pd.DataFrame([
        {'plant_id': feat['properties']['plant_id'], 'geometry': shape(feat['geometry'])}
        for feat in geojson_data['features']
    ])
    contours['plant_id'] = contours['plant_id'].astype(int)

    preview = np.array(Image.open("preview_rgb.png"))
    with open("block_boundaries.json", encoding="utf-8-sig") as f:
        block_boundaries = {k: np.array(v) for k, v in json.load(f).items()}
    return inventory, block_mapping, contours, preview, block_boundaries


def compute_block_health(block_plant_ids, inventory):
    df = inventory[inventory['plant_id'].isin(block_plant_ids)].copy()
    mean_ndvi, std_ndvi = df['ndvi'].mean(), df['ndvi'].std()
    df['ndvi_zscore'] = (df['ndvi'] - mean_ndvi) / (std_ndvi + 1e-6)

    def classify(z):
        if z < SOGLIA_STRESS:
            return 'stressed'
        elif z > SOGLIA_OTTIMALE:
            return 'vigorous'
        return 'normal'

    df['health_status'] = df['ndvi_zscore'].apply(classify)
    return df, mean_ndvi, std_ndvi


def explain_ndvi_range(mean_ndvi, min_ndvi, max_ndvi):
    if mean_ndvi > NDVI_HIGH_THRESHOLD:
        level, meaning = "high", "most plants in this block look vigorous and densely leaved"
    elif mean_ndvi < NDVI_LOW_THRESHOLD:
        level, meaning = "low", "many plants in this block show signs of reduced vigor or possible stress"
    else:
        level, meaning = "moderate", "this block shows a typical, mixed range — some healthier plants, some less so"
    return (f"NDVI values in this block range from **{min_ndvi:.2f}** to **{max_ndvi:.2f}** "
            f"(higher = denser, healthier-looking foliage). Compared to the rest of the nursery, "
            f"this is a **{level}** range — {meaning}.")


def compute_cwsi(t_plant, t_ref_healthy, t_dry):
    if pd.isna(t_plant):
        return None
    cwsi = (t_plant - t_ref_healthy) / (t_dry - t_ref_healthy)
    return float(np.clip(cwsi, 0, 1))


# --- Load data ---
inventory, block_mapping, contours, preview_img, block_boundaries = load_data()
T_DRY_GLOBAL = inventory['tir_mean'].quantile(0.99)

st.title("🌿 Nursery — Exploratory Demo")

available_blocks = sorted(block_mapping['blocco'].unique())
selected_block = st.selectbox("Select a block", available_blocks)

block_ids = block_mapping[block_mapping['blocco'] == selected_block]['plant_id']
block_health, mean_ndvi, std_ndvi = compute_block_health(block_ids, inventory)
block_contours = contours[contours['plant_id'].isin(block_ids)].merge(
    block_health[['plant_id', 'ndvi', 'ndvi_zscore', 'health_status', 'tir_mean']], on='plant_id'
)
low_confidence = std_ndvi < STD_MINIMA_AFFIDABILE
counts = block_health['health_status'].value_counts()

if "selected_plant" not in st.session_state:
    st.session_state.selected_plant = None

tab1, tab2 = st.tabs(["🗺️ Block Explorer", "🩺 Health & Verification"])

with tab1:
    st.metric("Total plants in this block", len(block_health))

    col_map, col_zoom = st.columns([1, 1])

    with col_map:
        st.subheader("Nursery overview")
        boundary = block_boundaries[selected_block] / ORTHO_SCALE
        bx = boundary[:, 1].tolist() + [boundary[0, 1]]
        by = boundary[:, 0].tolist() + [boundary[0, 0]]

        fig_overview = go.Figure()
        fig_overview.add_trace(go.Image(z=preview_img))
        fig_overview.add_trace(go.Scatter(
            x=bx, y=by, mode='lines', fill='toself',
            line=dict(color='white', width=2), fillcolor='rgba(255,255,255,0.15)',
            hoverinfo='skip', showlegend=False
        ))
        fig_overview.update_layout(
            height=550, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed')
        )
        st.plotly_chart(fig_overview, use_container_width=True)
        st.caption(f"**{selected_block}** — 🟢 {counts.get('vigorous', 0)} vigorous · "
                   f"🟡 {counts.get('normal', 0)} normal · 🔴 {counts.get('stressed', 0)} possibly stressed")

    with col_zoom:
        st.subheader(f"Zoom on {selected_block} — click a plant")
        pad = 0.15
        rows_full = block_boundaries[selected_block][:, 0]
        cols_full = block_boundaries[selected_block][:, 1]
        r_min = rows_full.min() - (rows_full.max() - rows_full.min()) * pad
        r_max = rows_full.max() + (rows_full.max() - rows_full.min()) * pad
        c_min = cols_full.min() - (cols_full.max() - cols_full.min()) * pad
        c_max = cols_full.max() + (cols_full.max() - cols_full.min()) * pad
        r_min_p, r_max_p = r_min / ORTHO_SCALE, r_max / ORTHO_SCALE
        c_min_p, c_max_p = c_min / ORTHO_SCALE, c_max / ORTHO_SCALE

        crop = preview_img[max(int(r_min_p), 0):int(r_max_p), max(int(c_min_p), 0):int(c_max_p)]

        fig_zoom = go.Figure()
        fig_zoom.add_trace(go.Image(z=crop))

        for status, color in STATUS_COLORS.items():
            subset = block_contours[block_contours['health_status'] == status]
            xs, ys = [], []
            cx, cy, ids, hover = [], [], [], []
            for _, row in subset.iterrows():
                coords = np.array(row['geometry'].exterior.coords) / ORTHO_SCALE
                xs.extend((coords[:, 0] - c_min_p).tolist() + [None])
                ys.extend((coords[:, 1] - r_min_p).tolist() + [None])
                c = row['geometry'].centroid
                cx.append(c.x / ORTHO_SCALE - c_min_p)
                cy.append(c.y / ORTHO_SCALE - r_min_p)
                ids.append(int(row['plant_id']))
                hover.append(f"Plant {int(row['plant_id'])} — {status}")

            fig_zoom.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', fill='toself',
                line=dict(color=color, width=1), fillcolor=color, opacity=0.5,
                hoverinfo='skip', showlegend=True, name=status.capitalize()
            ))
            if ids:
                fig_zoom.add_trace(go.Scatter(
                    x=cx, y=cy, mode='markers',
                    marker=dict(size=10, color=color, line=dict(color='black', width=1)),
                    customdata=ids, hovertext=hover, hoverinfo='text', showlegend=False
                ))

        fig_zoom.update_layout(
            height=550, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed'),
            legend=dict(orientation='h', y=-0.05)
        )

        event = st.plotly_chart(fig_zoom, use_container_width=True, on_select="rerun", key="zoom_chart")
        if event and event.get("selection", {}).get("points"):
            point = event["selection"]["points"][0]
            if "customdata" in point:
                st.session_state.selected_plant = point["customdata"][0]

    if st.session_state.selected_plant is not None:
        pid = st.session_state.selected_plant
        prow = block_health[block_health['plant_id'] == pid]
        if not prow.empty:
            prow = prow.iloc[0]
            st.markdown("---")
            st.subheader(f"Plant {pid}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("NDVI", f"{prow['ndvi']:.3f}")
            c2.metric("Status", prow['health_status'].capitalize())
            c3.metric("Leaf temperature", f"{prow['tir_mean']:.1f}°C" if pd.notna(prow['tir_mean']) else "n/a")
            c4.metric("NDVI z-score", f"{prow['ndvi_zscore']:.2f}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Canopy area", f"{prow['area_m2']:.2f} m²")

            if pd.notna(prow.get('altezza_media_m')):
                confidence_label = prow['confidenza_altezza'] if pd.notna(prow.get('confidenza_altezza')) else "n/a"
                height_display = f"{prow['altezza_media_m']:.2f} m"
                if prow['altezza_media_m'] < 0:
                    height_display += " ⚠️"
                c6.metric("Height", height_display, help=f"Confidence: {confidence_label}")
            else:
                c6.metric("Height", "n/a")

            c7.metric("Crown diameter", f"{prow['diametro_chioma_m']:.2f} m")

with tab2:
    st.subheader(f"Health overview — {selected_block}")
    st.write(explain_ndvi_range(mean_ndvi, block_health['ndvi'].min(), block_health['ndvi'].max()))

    if low_confidence:
        st.warning(
            "⚠️ **Low-confidence warning**: plants in this block have very similar NDVI values "
            "to each other. Small, perfectly normal differences might be flagged as a warning "
            "by mistake. We recommend using the temperature check below before drawing conclusions."
        )

    st.markdown("---")
    st.subheader("🌡️ Additional check: leaf temperature")
    st.write(
        "If you know the typical leaf temperature of a **healthy** plant of this species, "
        "enter it below. We'll use it to double-check the result with an independent method "
        "based on temperature (a water-stress index) instead of leaf color."
    )

    ref_temp = st.number_input(
        "Average leaf temperature of a healthy plant of this species (°C)",
        min_value=0.0, max_value=50.0, value=None, step=0.1,
        placeholder="e.g. 22.5"
    )

    if ref_temp is not None and st.session_state.selected_plant is not None:
        pid = st.session_state.selected_plant
        prow = block_health[block_health['plant_id'] == pid].iloc[0]

        cwsi = compute_cwsi(prow['tir_mean'], ref_temp, T_DRY_GLOBAL)

        if cwsi is None:
            st.info(f"No temperature data available for plant {pid} — cannot run this check.")
        else:
            cwsi_says_stressed = cwsi > CWSI_STRESS_THRESHOLD
            ndvi_says_stressed = prow['health_status'] == 'stressed'

            st.metric(f"CWSI for plant {pid}", f"{cwsi:.2f}", help="0 = no water stress, 1 = maximum water stress")

            if cwsi_says_stressed == ndvi_says_stressed:
                if ndvi_says_stressed:
                    st.error(f"✅ Both checks agree: plant {pid} shows signs of stress in both "
                             f"leaf color (NDVI) and temperature. This plant is worth a closer look.")
                else:
                    st.success(f"✅ Both checks agree: plant {pid} looks healthy, both in leaf "
                               f"color (NDVI) and temperature.")
            else:
                st.warning(
                    f"⚠️ The two checks disagree for plant {pid}: leaf color suggests "
                    f"**{'stress' if ndvi_says_stressed else 'no stress'}**, while temperature "
                    f"suggests **{'stress' if cwsi_says_stressed else 'no stress'}**. "
                    f"Possible reasons: the plant may be affected by something that shows up in "
                    f"one signal but not yet the other (e.g. early-stage stress visible in "
                    f"temperature before leaf color changes, or vice versa), or the reference "
                    f"temperature you entered may not be fully representative for this block."
                )
    elif ref_temp is not None:
        st.info("Select a plant in the 'Block Explorer' tab first to run this check on it.")
