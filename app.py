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

df_incidents, gdf_spatial = run_spatial_processing_pipeline()

# ==========================================
# BI-DIRECTIONAL SELECTION CONTROLLER
# ==========================================
if "selected_inc_id" not in st.session_state:
    st.session_state.selected_inc_id = None

CATEGORY_STYLES = {
    "Accident": {"color": "red", "icon": "car", "prefix": "fa"},
    "Road Works": {"color": "orange", "icon": "wrench", "prefix": "fa"},
    "Congestion": {"color": "amber", "icon": "clock-o", "prefix": "fa"},
    "Road Closure": {"color": "darkred", "icon": "ban", "prefix": "fa"},
    "General Alert": {"color": "blue", "icon": "exclamation-circle", "prefix": "fa"}
}

# 🎯 FIX ISSUE 3: Snap representation coordinates precisely onto the road lines
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
                # Interpolate finds the exact middle length along the line string path
                combined_line = inc_geom.unary_union
                if combined_line.geom_type in ['LineString', 'MultiLineString']:
                    rep_pt = combined_line.interpolate(0.5, normalized=True)
                else:
                    rep_pt = combined_line.representative_point()
            rep_points.append({'IncidentID': inc_id, 'lat': rep_pt.y, 'lng': rep_pt.x})
            
    if rep_points:
        df_rep = pd.DataFrame(rep_points)
        df_incidents = df_incidents.merge(df_rep, on='IncidentID', how='inner')

# 🎯 FIX ISSUE 4: Sync state index from Map interactions back to Table highlighting
default_selection_idx = []
if st.session_state.selected_inc_id in df_incidents['IncidentID'].values:
    matched_row_idx = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id].index[0]
    default_selection_idx = [int(matched_row_idx)]

# Grid Columns Layout Setup
col_table, col_map = st.columns([4, 5])

# 🎯 FIX ISSUE 2: Standardized headers to force side-by-side components parallel
with col_table:
    st.markdown("<div class='unified-header'><h3>📋 Traffic News Log</h3><p style='color:gray; font-size:14px; margin:0;'>Select a row here or use map markers directly.</p></div>", unsafe_allow_html=True)
    
    # 🎯 FIX ISSUE 1: Explicit column widths force scrollbar activation 
    selection = st.dataframe(
        df_incidents,
        use_container_width=True,
        hide_index=True,
        column_order=["IncidentID", "Category", "Location", "Details"],
        on_select="rerun",
        selection_mode="single-row",
        height=550, 
        selection_default={"selection": {"rows": default_selection_idx}},
        column_config={
            "IncidentID": st.column_config.TextColumn("Case ID", width=80),
            "Category": st.column_config.TextColumn("Type", width=110),
            "Location": st.column_config.TextColumn("Corridor Context", width=200), 
            "Details": st.column_config.TextColumn("Full Descriptive Summary Log View", width=600) # Expands past box boundary to generate scroll bars
        }
    )
    
    if selection and selection['selection']['rows']:
        row_idx = selection['selection']['rows'][0]
        table_selected_id = df_incidents.iloc[row_idx]['IncidentID']
        if st.session_state.selected_inc_id != table_selected_id:
            st.session_state.selected_inc_id = table_selected_id
            st.rerun()

with col_map:
    st.markdown("<div class='unified-header'><h3>🗺️ Map Visualization</h3><p style='color:gray; font-size:14px; margin:0;'>Click pins to expose corridor routes and full log summaries.</p></div>", unsafe_allow_html=True)
    
    m = folium.Map(location=[22.28552, 114.15769], zoom_start=12, tiles="cartodbpositron")
    
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
        
        # 🎯 FIX ISSUE 5: Append an auto-opening info box directly above the selected pin
        if is_current:
            popup_html = f"""
            <div style='font-family: Arial, sans-serif; font-size: 13px; width: 220px;'>
                <b>📍 {row['Location']}</b><br>
                <span style='color: #E63946;'>• {row['Category']}</span><br>
                <p style='margin-top:5px; color:#555;'>Click lower panel link to read full report.</p>
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
            
            total_bounds = matched_shapes_4326.total_bounds
            m.fit_bounds([[total_bounds[1], total_bounds[0]], [total_bounds[3], total_bounds[2]]])
    else:
        if 'lat' in df_incidents.columns and not df_incidents.empty:
            m.fit_bounds([[df_incidents['lat'].min(), df_incidents['lng'].min()], 
                          [df_incidents['lat'].max(), df_incidents['lng'].max()]])

    map_data = st_folium(m, width="100%", height=550, returned_objects=["last_object_clicked"])
    
    if map_data and map_data.get("last_object_clicked"):
        click_lat = map_data["last_object_clicked"]["lat"]
        click_lng = map_data["last_object_clicked"]["lng"]
        
        coordinate_match = df_incidents[
            (abs(df_incidents['lat'] - click_lat) < 0.0001) & 
            (abs(df_incidents['lng'] - click_lng) < 0.0001)
        ]
        if not coordinate_match.empty:
            new_map_selection = coordinate_match.iloc[0]['IncidentID']
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
