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


@st.cache_data
def compute_physical_thresholds(inventory):
    """25th/75th percentiles across the whole nursery, used to describe a
    plant's canopy area/diameter/height as small/medium/large."""
    return {
        'area_m2': (inventory['area_m2'].quantile(0.25), inventory['area_m2'].quantile(0.75)),
        'diametro_chioma_m': (inventory['diametro_chioma_m'].quantile(0.25),
                                inventory['diametro_chioma_m'].quantile(0.75)),
        'altezza_media_m': (inventory['altezza_media_m'].quantile(0.25),
                             inventory['altezza_media_m'].quantile(0.75)),
    }


def size_label(value, low, high):
    if pd.isna(value):
        return None
    if value < low:
        return "small"
    elif value > high:
        return "large"
    return "medium-sized"


def build_plant_summary(prow, thresholds):
    """Plain-language sentence describing this plant's canopy size and height."""
    area_label = size_label(prow['area_m2'], *thresholds['area_m2'])
    height_val = prow.get('altezza_media_m')
    height_label = None
    if pd.notna(height_val) and height_val >= 0:
        height_label = size_label(height_val, *thresholds['altezza_media_m'])

    if area_label and height_label:
        return (f"This plant has a **{area_label} canopy** and is **{height_label}** "
                f"compared to other plants in the nursery.")
    elif area_label:
        return (f"This plant has a **{area_label} canopy** compared to other plants in the "
                f"nursery. Height data isn't available for this plant.")
    return "Not enough data to describe this plant's overall size."


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


def explain_block_variability(mean_ndvi, std_ndvi, min_ndvi, max_ndvi, low_confidence):
    if mean_ndvi > NDVI_HIGH_THRESHOLD:
        level, meaning = "high", "most plants in this block look vigorous and densely leaved"
    elif mean_ndvi < NDVI_LOW_THRESHOLD:
        level, meaning = "low", "many plants in this block show signs of reduced vigor or possible stress"
    else:
        level, meaning = "moderate", "this block shows a typical, mixed range — some healthier plants, some less so"

    text = (f"Overall, this block sits in a **{level}** vigor range — {meaning}. "
            f"(Values observed span from {min_ndvi:.2f} to {max_ndvi:.2f} on our internal scale.)\n\n")

    if low_confidence:
        text += (
            "⚠️ **However, take individual differences within this block with caution.** "
            "The plants here are unusually similar to one another — there isn't much natural "
            "variation to compare against. This means that even small, perfectly normal "
            "differences between two plants might look like a warning sign when they aren't. "
            "We recommend using the temperature check below before concluding that any single "
            "plant is under stress."
        )
    else:
        text += (
            "✅ There's enough natural variation among plants in this block to make "
            "reliable comparisons — if a plant is flagged as stressed, it genuinely stands "
            "out from the others here."
        )
    return text


def compute_cwsi(t_plant, t_ref_healthy, t_dry):
    if pd.isna(t_plant):
        return None
    cwsi = (t_plant - t_ref_healthy) / (t_dry - t_ref_healthy)
    return float(np.clip(cwsi, 0, 1))


def build_zoom_figure(selected_block, block_boundaries, preview_img, color_by_status=False,
                       block_contours=None):
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

    fig = go.Figure()
    fig.add_trace(go.Image(z=crop))

    if color_by_status:
        groups = {status: block_contours[block_contours['health_status'] == status]
                  for status in STATUS_COLORS}
        color_map = STATUS_COLORS
    else:
        groups = {'plant': block_contours}
        color_map = {'plant': '#3498db'}

    for status, subset in groups.items():
        color = color_map[status]
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
            hover.append(f"Plant {int(row['plant_id'])}")

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', fill='toself',
            line=dict(color=color, width=1), fillcolor=color, opacity=0.5,
            hoverinfo='skip', showlegend=color_by_status, name=status.capitalize()
        ))
        if ids:
            fig.add_trace(go.Scatter(
                x=cx, y=cy, mode='markers',
                marker=dict(size=10, color=color, line=dict(color='black', width=1)),
                customdata=ids, hovertext=hover, hoverinfo='text', showlegend=False
            ))

    fig.update_layout(
        height=550, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed'),
        legend=dict(orientation='h', y=-0.05),
        clickmode='event+select'
    )
    return fig


def extract_selected_id(event):
    if event and event.get("selection", {}).get("points"):
        point = event["selection"]["points"][0]
        if "customdata" in point:
            cd = point["customdata"]
            return cd[0] if isinstance(cd, (list, tuple)) else cd
    return None


# --- Load data ---
inventory, block_mapping, contours, preview_img, block_boundaries = load_data()
T_DRY_GLOBAL = inventory['tir_mean'].quantile(0.99)
PHYSICAL_THRESHOLDS = compute_physical_thresholds(inventory)

st.title("🌿 Nursery — Exploratory Demo")

available_blocks = sorted(block_mapping['blocco'].unique())
selected_block = st.selectbox("Select a block", available_blocks)

block_ids = block_mapping[block_mapping['blocco'] == selected_block]['plant_id']
block_health, mean_ndvi, std_ndvi = compute_block_health(block_ids, inventory)
low_confidence = std_ndvi < STD_MINIMA_AFFIDABILE
counts = block_health['health_status'].value_counts()

block_contours = contours[contours['plant_id'].isin(block_ids)].merge(
    block_health[['plant_id', 'ndvi', 'ndvi_zscore', 'health_status', 'tir_mean']], on='plant_id'
)

if "plant_inventory" not in st.session_state:
    st.session_state.plant_inventory = None
if "plant_health" not in st.session_state:
    st.session_state.plant_health = None

tab1, tab2 = st.tabs(["🌳 Physical Inventory", "🩺 Health Status"])

# ============================================================
# TAB 1 — physical characteristics
# ============================================================
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
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed')
        )
        st.plotly_chart(fig_overview, use_container_width=True, key="overview_tab1")

    with col_zoom:
        st.subheader(f"Zoom on {selected_block} — click a plant")
        fig_zoom_inv = build_zoom_figure(selected_block, block_boundaries, preview_img,
                                           color_by_status=False, block_contours=block_contours)
        event_inv = st.plotly_chart(fig_zoom_inv, use_container_width=True,
                                      on_select="rerun", key="zoom_inventory")
        pid = extract_selected_id(event_inv)
        if pid is not None:
            st.session_state.plant_inventory = pid

    if st.session_state.plant_inventory is not None:
        pid = st.session_state.plant_inventory
        prow = inventory[inventory['plant_id'] == pid]
        if not prow.empty:
            prow = prow.iloc[0]
            st.markdown("---")
            st.subheader(f"Plant {pid} — physical characteristics")
            st.write(build_plant_summary(prow, PHYSICAL_THRESHOLDS))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Canopy area", f"{prow['area_m2']:.2f} m²")
                st.caption("The ground area covered by this plant's foliage, seen from above.")
            with c2:
                st.metric("Crown diameter", f"{prow['diametro_chioma_m']:.2f} m")
                st.caption("The width of the canopy — how wide the plant spreads.")
            with c3:
                if pd.notna(prow.get('altezza_media_m')) and prow['altezza_media_m'] >= 0:
                    confidence_label = prow['confidenza_altezza'] if pd.notna(prow.get('confidenza_altezza')) else "n/a"
                    st.metric("Height", f"{prow['altezza_media_m']:.2f} m",
                              help=f"Reliability of this measurement: {confidence_label}")
                    st.caption(f"Estimated from drone elevation data. Reliability: {confidence_label}.")
                else:
                    st.metric("Height", "n/a")
                    st.caption("Height couldn't be reliably measured for this plant "
                               "(e.g. due to limited elevation data coverage).")

# ============================================================
# TAB 2 — health status
# ============================================================
with tab2:
    st.subheader(f"Health overview — {selected_block}")
    st.markdown(explain_block_variability(mean_ndvi, std_ndvi,
                                            block_health['ndvi'].min(), block_health['ndvi'].max(),
                                            low_confidence))

    st.markdown("---")
    col_map2, col_zoom2 = st.columns([1, 1])
    with col_map2:
        st.subheader("Block status at a glance")
        st.write(f"🟢 Vigorous: {counts.get('vigorous', 0)}  ·  "
                 f"🟡 Normal: {counts.get('normal', 0)}  ·  "
                 f"🔴 Possibly stressed: {counts.get('stressed', 0)}")
        boundary = block_boundaries[selected_block] / ORTHO_SCALE
        bx = boundary[:, 1].tolist() + [boundary[0, 1]]
        by = boundary[:, 0].tolist() + [boundary[0, 0]]
        fig_overview2 = go.Figure()
        fig_overview2.add_trace(go.Image(z=preview_img))
        fig_overview2.add_trace(go.Scatter(
            x=bx, y=by, mode='lines', fill='toself',
            line=dict(color='white', width=2), fillcolor='rgba(255,255,255,0.15)',
            hoverinfo='skip', showlegend=False
        ))
        fig_overview2.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', autorange='reversed')
        )
        st.plotly_chart(fig_overview2, use_container_width=True, key="overview_tab2")

    with col_zoom2:
        st.subheader(f"Zoom on {selected_block} — click a plant")
        fig_zoom_health = build_zoom_figure(selected_block, block_boundaries, preview_img,
                                              color_by_status=True, block_contours=block_contours)
        event_health = st.plotly_chart(fig_zoom_health, use_container_width=True,
                                         on_select="rerun", key="zoom_health")
        pid = extract_selected_id(event_health)
        if pid is not None:
            st.session_state.plant_health = pid

    if st.session_state.plant_health is not None:
        pid = st.session_state.plant_health
        prow = block_health[block_health['plant_id'] == pid]
        if not prow.empty:
            prow = prow.iloc[0]
            st.markdown("---")
            st.subheader(f"Plant {pid} — health status")
            c1, c2 = st.columns(2)
            c1.metric("Status", prow['health_status'].capitalize())
            c2.metric("Leaf temperature", f"{prow['tir_mean']:.1f}°C" if pd.notna(prow['tir_mean']) else "n/a")

            st.markdown("---")
            st.subheader("🌡️ Additional check: leaf temperature")
            st.write(
                "If you know the typical leaf temperature of a **healthy** plant of this "
                "species, enter it below. We'll use it to double-check the result with an "
                "independent method based on temperature, instead of leaf color."
            )
            ref_temp = st.number_input(
                "Average leaf temperature of a healthy plant of this species (°C)",
                min_value=0.0, max_value=50.0, value=None, step=0.1,
                placeholder="e.g. 22.5", key="ref_temp_input"
            )

            if ref_temp is not None:
                cwsi = compute_cwsi(prow['tir_mean'], ref_temp, T_DRY_GLOBAL)
                if cwsi is None:
                    st.info(f"No temperature data available for plant {pid} — cannot run this check.")
                else:
                    cwsi_says_stressed = cwsi > CWSI_STRESS_THRESHOLD
                    ndvi_says_stressed = prow['health_status'] == 'stressed'
                    st.metric("Water stress index (CWSI)", f"{cwsi:.2f}",
                              help="0 = no water stress, 1 = maximum water stress")

                    if cwsi_says_stressed == ndvi_says_stressed:
                        if ndvi_says_stressed:
                            st.error("✅ Both checks agree: this plant shows signs of stress "
                                     "in both leaf color and temperature. Worth a closer look.")
                        else:
                            st.success("✅ Both checks agree: this plant looks healthy, both "
                                       "in leaf color and temperature.")
                    else:
                        st.warning(
                            f"⚠️ The two checks disagree: leaf color suggests "
                            f"**{'stress' if ndvi_says_stressed else 'no stress'}**, while "
                            f"temperature suggests **{'stress' if cwsi_says_stressed else 'no stress'}**. "
                            f"This might mean the stress is only just beginning to show in one "
                            f"signal but not the other yet, or that the reference temperature "
                            f"you entered isn't fully representative for this block."
                        )
    else:
        st.info("Click a plant on the map above to see its health status and run the temperature check.")
