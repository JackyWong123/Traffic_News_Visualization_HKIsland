# app.py
import os
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from traffic_processor import TrafficIncidentEngine

st.set_page_config(layout="wide", page_title="HKI Traffic Dashboard")

# Custom CSS styling to force headers to align perfectly and clean up dataframe borders
st.markdown("""
    <style>
        div[data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .unified-header {
            margin-bottom: 24px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Traffic News Visualization for Hong Kong Island")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def run_spatial_processing_pipeline():
    engine = TrafficIncidentEngine(
        road_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_centerline.zip"),
        boundary_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_boundary.zip"),
        building_path=os.path.join(BASE_DIR, "iG1000_HKI_FGDB", "hki_building.zip"),
        xml_path=os.path.join(BASE_DIR, "combined.xml"),
        distance_threshold=500
    )
    return engine.process_active_incidents()

raw_incidents, gdf_spatial = run_spatial_processing_pipeline()

# ==========================================
# STATIC DATA PREPARATION (LOCKS ROW INDICES)
# ==========================================
# We process coordinates immediately to ensure row index integers never shift
if "selected_inc_id" not in st.session_state:
    st.session_state.selected_inc_id = None

df_incidents = raw_incidents.copy()

if not df_incidents.empty and 'lat' not in df_incidents.columns:
    gdf_4326 = gdf_spatial.to_crs(epsg=4326)
    rep_points = []
    
    for inc_id in df_incidents['IncidentID']:
        inc_geom = gdf_4326[gdf_4326['IncidentID'] == inc_id]
        if not inc_geom.empty:
            point_feats = inc_geom[inc_geom.geometry.geom_type == 'Point']
            if not point_feats.empty:
                rep_pt = point_feats.iloc[0]['geometry']
            else:
                combined_line = inc_geom.unary_union
                if combined_line.geom_type in ['LineString', 'MultiLineString']:
                    rep_pt = combined_line.interpolate(0.5, normalized=True)
                else:
                    rep_pt = combined_line.representative_point()
            rep_points.append({'IncidentID': inc_id, 'lat': rep_pt.y, 'lng': rep_pt.x})
            
    if rep_points:
        df_rep = pd.DataFrame(rep_points)
        df_incidents = df_incidents.merge(df_rep, on='IncidentID', how='inner')

# Stable index resolution for cross-component highlighting
default_selection_idx = []
if st.session_state.selected_inc_id in df_incidents['IncidentID'].values:
    matched_row_idx = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id].index[0]
    default_selection_idx = [int(matched_row_idx)]

CATEGORY_STYLES = {
    "Accident": {"color": "red", "icon": "car", "prefix": "fa"},
    "Road Works": {"color": "orange", "icon": "wrench", "prefix": "fa"},
    "Congestion": {"color": "amber", "icon": "clock-o", "prefix": "fa"},
    "Road Closure": {"color": "darkred", "icon": "ban", "prefix": "fa"},
    "General Alert": {"color": "blue", "icon": "exclamation-circle", "prefix": "fa"}
}

# ==========================================
# INTERACTIVE SCREEN LAYOUT GENERATION
# ==========================================
col_table, col_map = st.columns([4, 5])

with col_table:
    st.markdown("<div class='unified-header'><h3>Traffic News Log</h3><p style='color:gray; font-size:14px; margin:0;'>Click anywhere on a row to isolate the incident footprint.</p></div>", unsafe_allow_html=True)
    
    selection = st.dataframe(
        df_incidents,
        use_container_width=True,
        hide_index=True,
        column_order=["IncidentID", "Category", "Location", "Details"],
        on_select="rerun",
        selection_mode="single-row",
        height=550, 
        selection_default={"selection": {"rows": default_selection_idx}},
        key=f"grid_sync_{st.session_state.selected_inc_id}", # Unified state re-render flag
        column_config={
            "IncidentID": st.column_config.TextColumn("Case ID", width=80),
            "Category": st.column_config.TextColumn("Type", width=110),
            "Location": st.column_config.TextColumn("Corridor Context", width=180), 
            "Details": st.column_config.TextColumn("Summary Log View (Click row to read full text below)", width=600) 
        }
    )
    
    if selection and selection['selection']['rows']:
        row_idx = selection['selection']['rows'][0]
        table_selected_id = df_incidents.iloc[row_idx]['IncidentID']
        if st.session_state.selected_inc_id != table_selected_id:
            st.session_state.selected_inc_id = table_selected_id
            st.rerun()

with col_map:
    st.markdown("<div class='unified-header'><h3>Map Visualization</h3><p style='color:gray; font-size:14px; margin:0;'>Click pins to expose corridor routes and full log summaries.</p></div>", unsafe_allow_html=True)
    
    # 🎯 FIX POP-UP: Determine coordinates BEFORE generating the folium Map canvas
    map_center = [22.28552, 114.15769]
    zoom_level = 11
    
    if st.session_state.selected_inc_id:
        active_rows = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id]
        if not active_rows.empty:
            map_center = [active_rows.iloc[0]['lat'], active_rows.iloc[0]['lng']]
            zoom_level = 15
            
    m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="cartodbpositron")
    
    # Draw all elements onto map container
    for _, row in df_incidents.iterrows():
        is_current = (row['IncidentID'] == st.session_state.selected_inc_id)
        style = CATEGORY_STYLES.get(row['Category'], CATEGORY_STYLES["General Alert"])
        
        bg_color = "darkpurple" if is_current else style["color"]
        icon_color = "white"
        
        marker = folium.Marker(
            location=[row['lat'], row['lng']],
            icon=folium.Icon(color=bg_color, icon_color=icon_color, icon=style["icon"], prefix=style["prefix"]),
            tooltip=f"ID: {row['IncidentID']} - {row['Location']}"
        )
        
        if is_current:
            popup_html = f"""
            <div style='font-family: Arial, sans-serif; font-size: 13px; width: 220px; padding: 5px;'>
                <b style='color: #333;'>📍 {row['Location']}</b><br>
                <span style='color: #E63946; font-weight: bold;'>• {row['Category']}</span><br>
                <p style='margin: 5px 0 0 0; color: #666; font-size: 11px;'>Full description loaded below log table.</p>
            </div>
            """
            folium.Popup(popup_html, show=True, max_width=250).add_to(marker)
            
        marker.add_to(m)
        
    if st.session_state.selected_inc_id:
        matched_shapes = gdf_spatial[gdf_spatial['IncidentID'] == st.session_state.selected_inc_id]
        if not matched_shapes.empty:
            matched_shapes_4326 = matched_shapes.to_crs(epsg=4326)
            for _, row in matched_shapes_4326.iterrows():
                geom = row['geometry']
                if geom is None or geom.geom_type == 'Point': continue
                folium.GeoJson(geom, style_function=lambda x: {'color': '#E63946', 'weight': 7, 'opacity': 0.9}).add_to(m)

    map_data = st_folium(m, width="100%", height=550, returned_objects=["last_object_clicked"])
    
    # 🎯 FIX SYNC: Calculate absolute nearest neighbor to bypass float rounding bugs
    if map_data and map_data.get("last_object_clicked") and not df_incidents.empty:
        click_lat = map_data["last_object_clicked"]["lat"]
        click_lng = map_data["last_object_clicked"]["lng"]
        
        # Vectorized distance formula (hypotenuse delta coordinates)
        spatial_distances = ((df_incidents['lat'] - click_lat)**2 + (df_incidents['lng'] - click_lng)**2)
        true_closest_idx = spatial_distances.idxmin()
        
        # Verify click lands close to a pin (within roughly 400 meters)
        if spatial_distances[true_closest_idx] < 0.00015:
            new_map_selection = df_incidents.loc[true_closest_idx, 'IncidentID']
            if st.session_state.selected_inc_id != new_map_selection:
                st.session_state.selected_inc_id = new_map_selection
                st.rerun()

# ==========================================
# LOWER PANEL FULL TEXT DISCOVERY GRID
# ==========================================
if st.session_state.selected_inc_id:
    active_record = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id].iloc[0]
    st.write("---")
    st.markdown(f"### Full Traffic News (Case ID: {active_record['IncidentID']})")
    st.info(f"**Incident Category Classification:** {active_record['Category']}\n\n**Location Corridor:** {active_record['Location']}\n\n**Incident Description:** {active_record['Details']}")
