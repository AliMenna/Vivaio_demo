import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from PIL import Image
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Nursery - Exploratory Demo", layout="wide")

ORTHO_SCALE = 4.228  # scale factor: full-resolution pixels / preview pixels

# --- Calibrated parameters (see notebook analysis) ---
STD_MINIMA_AFFIDABILE = 0.03
NDVI_HIGH_THRESHOLD = 0.929   # 75th percentile, whole nursery
NDVI_LOW_THRESHOLD = 0.752    # 25th percentile, whole nursery
SOGLIA_STRESS = -1.5
SOGLIA_OTTIMALE = 1.0

STATUS_COLORS = {'vigorous': '#2ecc71', 'normal': '#f1c40f', 'stressed': '#e74c3c'}


@st.cache_data
def load_data():
    inventory = pd.read_csv("inventario_app.csv")
    block_mapping = pd.read_csv("mappatura_blocchi.csv")
    contours = gpd.read_file("contorni_tutte_piante.geojson")
    preview = Image.open("preview_rgb.png")
    with open("block_boundaries.json") as f:
        block_boundaries = json.load(f)
    return inventory, block_mapping, contours, preview, block_boundaries


def compute_block_health(block_plant_ids, inventory):
    """Compute NDVI-based health status for each plant in a block, using
    z-score relative to the block's own mean/std."""
    df = inventory[inventory['plant_id'].isin(block_plant_ids)].copy()
    mean_ndvi = df['ndvi'].mean()
    std_ndvi = df['ndvi'].std()

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
    """Plain-language explanation of where this block's NDVI sits."""
    if mean_ndvi > NDVI_HIGH_THRESHOLD:
        level = "high"
        meaning = "most plants in this block look vigorous and densely leaved"
    elif mean_ndvi < NDVI_LOW_THRESHOLD:
        level = "low"
        meaning = "many plants in this block show signs of reduced vigor or possible stress"
    else:
        level = "moderate"
        meaning = "this block shows a typical, mixed range — some healthier plants, some less so"

    return (f"NDVI values in this block range from **{min_ndvi:.2f}** to **{max_ndvi:.2f}** "
            f"(higher values mean denser, healthier-looking foliage). "
            f"Compared to the rest of the nursery, this is a **{level}** range — {meaning}.")


# --- App ---
st.title("🌿 Nursery — Exploratory Demo")

inventory, block_mapping, contours, preview, block_boundaries = load_data()

st.header("Block Health Overview")

available_blocks = sorted(block_mapping['blocco'].unique())
selected_block = st.selectbox("Select a block", available_blocks)

block_ids = block_mapping[block_mapping['blocco'] == selected_block]['plant_id']
block_health, mean_ndvi, std_ndvi = compute_block_health(block_ids, inventory)
block_health = block_health.merge(inventory[['plant_id', 'tir_mean']], on='plant_id', how='left')

low_confidence = std_ndvi < STD_MINIMA_AFFIDABILE
range_explanation = explain_ndvi_range(mean_ndvi, block_health['ndvi'].min(), block_health['ndvi'].max())
counts = block_health['health_status'].value_counts()

alert_text = (
    "Low-confidence: plants in this block have very similar NDVI values, "
    "making it harder to reliably tell which ones truly stand out."
) if low_confidence else "Confidence: normal — the block shows enough natural variation to trust this comparison."

block_hover_text = (
    f"<b>{selected_block}</b><br>"
    f"{len(block_health)} plants<br>"
    f"Vigorous: {counts.get('vigorous', 0)} | Normal: {counts.get('normal', 0)} | "
    f"Stressed: {counts.get('stressed', 0)}<br>"
    f"{alert_text}"
)

# --- Build the figure ---
img_array = np.array(preview)
fig = go.Figure()
fig.add_trace(go.Image(z=img_array))

# Block boundary: semi-transparent outline + a hoverable centroid marker with aggregate info
boundary = np.array(block_boundaries[selected_block]) / ORTHO_SCALE  # (row, col) pairs
boundary_x = boundary[:, 1].tolist() + [boundary[0, 1]]
boundary_y = boundary[:, 0].tolist() + [boundary[0, 0]]

fig.add_trace(go.Scatter(
    x=boundary_x, y=boundary_y, mode='lines', fill='toself',
    line=dict(color='rgba(255,255,255,0.8)', width=2),
    fillcolor='rgba(255,255,255,0.08)',
    hoverinfo='skip', showlegend=False
))
fig.add_trace(go.Scatter(
    x=[boundary[:, 1].mean()], y=[boundary[:, 0].mean()],
    mode='markers', marker=dict(size=18, color='white', symbol='star',
                                  line=dict(color='black', width=1)),
    hovertemplate=block_hover_text + "<extra></extra>",
    showlegend=False
))

# Individual plant contours, colored by status, each with its own hover
block_contours = contours[contours['plant_id'].isin(block_ids)].merge(
    block_health[['plant_id', 'ndvi', 'ndvi_zscore', 'health_status', 'tir_mean']], on='plant_id'
)

for status, color in STATUS_COLORS.items():
    subset = block_contours[block_contours['health_status'] == status]
    xs, ys = [], []
    for _, row in subset.iterrows():
        coords = np.array(row.geometry.exterior.coords) / ORTHO_SCALE
        xs.extend(coords[:, 0].tolist() + [None])
        ys.extend(coords[:, 1].tolist() + [None])

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines', fill='toself',
        line=dict(color=color, width=1), fillcolor=color, opacity=0.55,
        hoverinfo='skip', showlegend=True, name=status.capitalize()
    ))

    if len(subset) > 0:
        hover = []
        for _, row in subset.iterrows():
            temp_str = f"{row['tir_mean']:.1f}°C" if pd.notna(row['tir_mean']) else "not available"
            hover.append(
                f"<b>Plant {row['plant_id']}</b><br>NDVI: {row['ndvi']:.3f}<br>"
                f"Status: {status}<br>Leaf temperature: {temp_str}"
            )
        centroids = subset.geometry.centroid
        fig.add_trace(go.Scatter(
            x=(centroids.x / ORTHO_SCALE), y=(centroids.y / ORTHO_SCALE),
            mode='markers', marker=dict(size=6, color=color, opacity=0),
            hovertext=hover, hoverinfo='text', showlegend=False
        ))

fig.update_layout(
    height=700, margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed'),
    legend=dict(orientation='h', y=-0.02)
)

st.plotly_chart(fig, use_container_width=True)
st.write(range_explanation)
if low_confidence:
    st.warning(f"⚠️ {alert_text} We recommend using the additional temperature check below.")
