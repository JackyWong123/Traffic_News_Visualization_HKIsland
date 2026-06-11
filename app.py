# app.py
import os
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from traffic_processor import TrafficIncidentEngine

st.set_page_config(layout="wide", page_title="HKI Traffic Dashboard")

# Custom CSS styling to align headers and style our scrollable interactive cards
st.markdown("""
    <style>
        div[data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .unified-header {
            margin-bottom: 24px;
            height: 70px;
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
# SELECTION MEMORY & VIEWPORT ANCHORS
# ==========================================
if "selected_inc_id" not in st.session_state:
    st.session_state.selected_inc_id = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [22.28552, 114.15769]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 12

df_incidents = raw_incidents.copy()

# Lock coordinates onto rows
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
    # Fix 1: Text-only header without emojis
    st.markdown("<div class='unified-header'><h3>Traffic News Log</h3><p style='color:gray; font-size:14px; margin:0;'>Click 'Focus on Map' inside any log card to isolate its corridor footprint.</p></div>", unsafe_allow_html=True)
    
    # Fix 2 & 4: Custom Scrollable Feed Container with Text Wrapping and Row Selection
    with st.container(height=550):
        for _, row in df_incidents.iterrows():
            is_current = (row['IncidentID'] == st.session_state.selected_inc_id)
            
            # Change outline border and background color dynamically based on active state
            card_border = "border: 2px solid #E63946; background-color: #FFF5F5;" if is_current else "border: 1px solid #E0E0E0; background-color: #FAFAFA;"
            badge_color = CATEGORY_STYLES.get(row['Category'], {"color": "gray"})["color"]
            if badge_color == "amber": badge_color = "#FFB000"
            
            # Clean HTML template to force text wrapping
            card_template = f"""
            <div style="{card_border} padding: 14px; border-radius: 8px; margin-bottom: 12px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: bold; font-size: 14px; color: #111;">Case ID: {row['IncidentID']}</span>
                    <span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; text-transform: uppercase;">{row['Category']}</span>
                </div>
                <div style="font-weight: bold; color: #444; font-size: 13px; margin-bottom: 6px;">📍 {row['Location']}</div>
                <div style="font-size: 13px; color: #555; line-height: 1.4; word-wrap: break-word; white-space: normal;">{row['Details']}</div>
            </div>
            """
            st.markdown(card_template, unsafe_allow_html=True)
            
            # Simple interaction button right under the text card
            if st.button("Focus on Map", key=f"select_btn_{row['IncidentID']}", use_container_width=True, type="primary" if is_current else "secondary"):
                st.session_state.selected_inc_id = row['IncidentID']
                st.session_state.map_center = [row['lat'], row['lng']]
                st.session_state.map_zoom = 16
                st.rerun()

with col_map:
    # Fix 1: Text-only header without emojis
    st.markdown("<div class='unified-header'><h3>Map Visualization</h3><p style='color:gray; font-size:14px; margin:0;'>Click pins directly to cross-examine and sync the traffic logs.</p></div>", unsafe_allow_html=True)
    
    # Generate map object using our stabilized layout coordinates
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles="cartodbpositron")
    
    # Render map markers
    for _, row in df_incidents.iterrows():
        is_current = (row['IncidentID'] == st.session_state.selected_inc_id)
        style = CATEGORY_STYLES.get(row['Category'], CATEGORY_STYLES["General Alert"])
        
        bg_color = "darkpurple" if is_current else style["color"]
        
        popup_html = f"""
        <div style='font-family: Arial, sans-serif; font-size: 13px; width: 210px; padding: 3px;'>
            <b style='color: #222;'>📍 {row['Location']}</b><br>
            <span style='color: #E63946; font-weight: bold;'>• {row['Category']}</span><br>
            <p style='margin: 5px 0 0 0; color: #555; font-size: 11px; line-height:1.3;'>Full text loaded underneath log column feed.</p>
        </div>
        """
        
        folium.Marker(
            location=[row['lat'], row['lng']],
            icon=folium.Icon(color=bg_color, icon_color="white", icon=style["icon"], prefix=style["prefix"]),
            tooltip=f"ID: {row['IncidentID']} - {row['Location']}",
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)
        
    # Project linear vector links for the selected incident
    if st.session_state.selected_inc_id:
        matched_shapes = gdf_spatial[gdf_spatial['IncidentID'] == st.session_state.selected_inc_id]
        if not matched_shapes.empty:
            matched_shapes_4326 = matched_shapes.to_crs(epsg=4326)
            for _, row in matched_shapes_4326.iterrows():
                geom = row['geometry']
                if geom is None or geom.geom_type == 'Point': continue
                folium.GeoJson(geom, style_function=lambda x: {'color': '#E63946', 'weight': 7, 'opacity': 0.9}).add_to(m)

    # Render widget and track viewport parameters
    map_data = st_folium(m, width="100%", height=550, returned_objects=["last_object_clicked", "center", "zoom"])
    
    # Fix 3 & 5: Save active zoom positions dynamically to prevent rendering layout tears
    if map_data:
        if map_data.get("center") and map_data.get("zoom"):
            current_lat = map_data["center"]["lat"]
            current_lng = map_data["center"]["lng"]
            current_zoom = map_data["zoom"]
            
            # Update state anchors silently behind the scenes
            st.session_state.map_center = [current_lat, current_lng]
            st.session_state.map_zoom = current_zoom

    # Handle map pin click events
    if map_data and map_data.get("last_object_clicked") and not df_incidents.empty:
        click_lat = map_data["last_object_clicked"]["lat"]
        click_lng = map_data["last_object_clicked"]["lng"]
        
        # Determine closest geographic coordinate match
        spatial_distances = ((df_incidents['lat'] - click_lat)**2 + (df_incidents['lng'] - click_lng)**2)
        true_closest_idx = spatial_distances.idxmin()
        
        if spatial_distances[true_closest_idx] < 0.0005:
            new_map_selection = df_incidents.loc[true_closest_idx, 'IncidentID']
            if st.session_state.selected_inc_id != new_map_selection:
                st.session_state.selected_inc_id = new_map_selection
                # Freeze viewport exactly where the map is right now to anchor the popup perfectly
                st.session_state.map_center = [click_lat, click_lng]
                st.rerun()

# ==========================================
# LOWER PANEL FULL TEXT DISCOVERY GRID
# ==========================================
if st.session_state.selected_inc_id:
    active_record = df_incidents[df_incidents['IncidentID'] == st.session_state.selected_inc_id].iloc[0]
    st.write("---")
    st.markdown(f"### Full Traffic News (Case ID: {active_record['IncidentID']})")
    st.info(f"**Incident Category Classification:** {active_record['Category']}\n\n**Location Corridor:** {active_record['Location']}\n\n**Incident Description:** {active_record['Details']}")
