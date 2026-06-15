# traffic_processor.py
import xml.etree.ElementTree as ET
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

class TrafficIncidentEngine:
    def __init__(self, road_path, boundary_path, building_path, xml_path, distance_threshold=500):
        self.road_path = road_path
        self.boundary_path = boundary_path
        self.building_path = building_path  
        self.xml_path = xml_path
        self.distance_threshold = distance_threshold
        
        self.road_df = None
        self.landmark_cache = {}
        self.road_names_cache = set()
        
    def initialize_spatial_basemaps(self):
        """Loads GIS layers directly out of compressed zip files."""
        self.road_df = gpd.read_file(f"zip://{self.road_path}")
        boundary_df = gpd.read_file(f"zip://{self.boundary_path}")
        building_df = gpd.read_file(f"zip://{self.building_path}") 
        
        self.road_df.columns = self.road_df.columns.str.upper()
        boundary_df.columns = boundary_df.columns.str.upper()
        building_df.columns = building_df.columns.str.upper()
        
        self.road_df = self.road_df.set_geometry("GEOMETRY").set_crs(epsg=2326, allow_override=True)
        boundary_df = boundary_df.set_geometry("GEOMETRY").set_crs(epsg=2326, allow_override=True)
        building_df = building_df.set_geometry("GEOMETRY").set_crs(epsg=2326, allow_override=True)
            
        for _, feature in boundary_df.iterrows():
            geom = feature['GEOMETRY']
            if geom is None: continue
            centroid = geom.centroid
            
            for field in ['ENGLISHSITENAME', 'ENGLISHBUILDINGNAME']:
                if field in boundary_df.columns and pd.notna(feature[field]):
                    val = str(feature[field]).upper().strip()
                    if len(val) > 3 and val not in ["NAN", "<NA>"]: 
                        self.landmark_cache[val] = centroid

        for _, feature in building_df.iterrows():
            geom = feature['GEOMETRY']
            if geom is None: continue
            centroid = geom.centroid
            
            for field in ['BUILDINGNAME', 'ENGLISHBUILDINGNAME', 'BUILDING_EN']:
                if field in building_df.columns and pd.notna(feature[field]):
                    val = str(feature[field]).upper().strip()
                    if len(val) > 3 and val not in ["NAN", "<NA>"] and val not in self.landmark_cache: 
                        self.landmark_cache[val] = centroid

        if 'STREET_ENAME' in self.road_df.columns:
            self.road_names_cache = set(self.road_df['STREET_ENAME'].dropna().str.upper().str.strip())

    def classify_incident(self, content_upper):
        if any(kw in content_upper for kw in ["ACCIDENT", "COLLISION", "CAR CRASH"]):
            return "Accident"
        elif any(kw in content_upper for kw in ["ROAD WORKS", "ROADWORKS", "MAINTENANCE", "REPAIR"]):
            return "Road Works"
        elif any(kw in content_upper for kw in ["TRAFFIC QUEUE", "CONGESTION", "BUSY", "SLOW TRAFFIC"]):
            return "Congestion"
        elif any(kw in content_upper for kw in ["CLOSED", "POLICE INVESTIGATION", "BLOCKAGE"]):
            return "Road Closure"
        else:
            return "General Alert"

    def filter_directional_bounds(self, matched_roads, bound_compass):
        """Topological Anti-Vector Engine: Drops split highway lanes traveling the wrong way."""
        if not bound_compass or matched_roads.empty:
            return matched_roads
            
        # Scan for the correct column name safely
        dir_col = None
        for col in matched_roads.columns:
            if any(k in col for k in ['TRAVEL', 'DIR', 'TRAFFIC']):
                dir_col = col
                break
                
        # 🎯 DYNAMIC ENVIRONMENT ONSHORE AXIS DETECTOR
        # Inspects library definitions to see if the cloud environment flipped coordinate order arrays
        is_northing_first = False
        if matched_roads.crs and hasattr(matched_roads.crs, 'axis_info'):
            try:
                if matched_roads.crs.axis_info[0].direction.lower() == 'north':
                    is_northing_first = True
            except:
                pass
                
        valid_indices = []
        for idx, road_feat in matched_roads.iterrows():
            geom = road_feat['GEOMETRY']
            if geom is None: continue
            
            dir_val = "1"
            if dir_col and pd.notna(road_feat[dir_col]):
                dir_val = str(road_feat[dir_col]).strip().split('.')[0]
                
            if dir_val == '3':
                # Extract coordinate sequence pairs out of geometry layers safely
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                elif geom.geom_type == 'MultiLineString' and not geom.is_empty:
                    coords = list(max(geom.geoms, key=lambda l: l.length).coords)
                else:
                    valid_indices.append(idx)
                    continue
                    
                if len(coords) < 2:
                    valid_indices.append(idx)
                    continue
                
                first_node = coords[0]
                last_node = coords[-1]
                
                if is_northing_first:
                    # Cloud server sequence standard rules tracking
                    n_start, e_start = first_node[0], first_node[1]
                    n_end, e_end = last_node[0], last_node[1]
                else:
                    # Traditional workstation desktop workspace standards rules tracking
                    e_start, n_start = first_node[0], first_node[1]
                    e_end, n_end = last_node[0], last_node[1]
                    
                dx = e_end - e_start # True Easting trajectory vector change
                dy = n_end - n_start # True Northing trajectory vector change
                
                # ANTI-VECTOR CRITERIA RANGE SELECTOR
                keep_segment = True
                if bound_compass == "WEST" and dx > 10: keep_segment = False
                if bound_compass == "EAST" and dx < -10: keep_segment = False
                if bound_compass == "SOUTH" and dy > 10: keep_segment = False
                if bound_compass == "NORTH" and dy < -10: keep_segment = False
                
                if keep_segment:
                    valid_indices.append(idx)
            else:
                # Code 1: Combined Undivided Highway/Street Asset. Must be retained.
                valid_indices.append(idx)
                
        # Allocation safeguards enforce fallback overrides if total loss calculation triggers
        if 0 < len(valid_indices) < len(matched_roads):
            return matched_roads.loc[valid_indices]
            
        return matched_roads

    @staticmethod
    def _get_xml_text(message_element, tag_name):
        for child in message_element:
            if child.tag.split('}')[-1] == tag_name:
                return child.text.strip() if child.text else ""
        return ""

    def process_active_incidents(self):
        if self.road_df is None:
            self.initialize_spatial_basemaps()

        with open(self.xml_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        message_blocks = re.findall(r'<message>.*?</message>', full_text, re.DOTALL)
        incident_records = []
        spatial_features_list = []

        for block in message_blocks:
            try:
                root_snippet = ET.fromstring(f"<root>{block}</root>")
                message = root_snippet.find('message')
            except: continue
            if message is None: continue

            inc_id = self._get_xml_text(message, 'ID')
            status = self._get_xml_text(message, 'INCIDENT_STATUS_EN').upper().strip()
            location_en = self._get_xml_text(message, 'LOCATION_EN').upper().strip()
            content_en = self._get_xml_text(message, 'CONTENT_EN').strip()
            content_upper = content_en.upper()

            if status == "CLOSED" or not location_en: 
                continue
                
            active_text_block = content_upper.split("RESUMED NORMAL")[0]
            category = self.classify_incident(content_upper)

            # 🎯 EXTENDED WHOLE HONG KONG GEOGRAPHIC KEYWORD POOL MATRICES
            bound_compass = None
            text_pool = content_upper + " " + location_en
            if any(kw in text_pool for kw in ["WEST BOUND", "WESTBOUND", "CENTRAL BOUND", "CENTRAL-BOUND", "KENNEDY TOWN BOUND", "SHEUNG WAN BOUND", "TO CENTRAL", "TUEN MUN BOUND", "YUEN LONG BOUND", "TSUEN WAN BOUND"]):
                bound_compass = "WEST"
            elif any(kw in text_pool for kw in ["EAST BOUND", "EASTBOUND", "CHAI WAN BOUND", "EASTERN BOUND", "CAUSEWAY BAY BOUND", "QUARRY BAY BOUND", "NORTH POINT BOUND", "TO CHAI WAN", "SAI KUNG BOUND", "MA ON SHAN BOUND", "TAI PO BOUND"]):
                bound_compass = "EAST"
            elif any(kw in text_pool for kw in ["SOUTH BOUND", "SOUTHBOUND", "ABERDEEN BOUND", "STANLEY BOUND", "REPULSE BAY BOUND", "WONG CHUK HANG BOUND"]):
                bound_compass = "SOUTH"
            elif any(kw in text_pool for kw in ["NORTH BOUND", "NORTHBOUND", "KOWLOON BOUND", "CROSS HARBOUR", "SHATIN BOUND", "SHA TIN BOUND", "FANLING BOUND", "SHEUNG SHUI BOUND", "KWUN TONG BOUND"]):
                bound_compass = "NORTH"

            main_road = None
            if location_en in self.road_names_cache:
                main_road = location_en
            else:
                for r in sorted(list(self.road_names_cache), key=len, reverse=True):
                    if r in location_en:
                        main_road = r
                        break

            if location_en != "BUSY ROAD SECTIONS" and not main_road:
                continue

            # TRACK B: BROADCAST MODE
            if location_en == "BUSY ROAD SECTIONS":
                sorted_roads = sorted(list(self.road_names_cache), key=len, reverse=True)
                text_to_scan = active_text_block
                matched_any_hki_road = False
                
                for cached_road in sorted_roads:
                    if re.search(r'\b' + re.escape(cached_road) + r'\b', text_to_scan):
                        text_to_scan = text_to_scan.replace(cached_road, " __SPATIAL_MATCH__ ")
                        
                        matched_roads = self.road_df[self.road_df['STREET_ENAME'] == cached_road]
                        matched_roads = self.filter_directional_bounds(matched_roads, bound_compass)
                        
                        if not matched_roads.empty:
                            for _, road_feat in matched_roads.iterrows():
                                matched_any_hki_road = True
                                spatial_features_list.append({
                                    'IncidentID': inc_id, 'RoadName': cached_road, 
                                    'RouteID': str(road_feat.get('ROUTE_ID', 'UNKNOWN')), 'geometry': road_feat['GEOMETRY']
                                })
                
                if matched_any_hki_road:
                    incident_records.append({'IncidentID': inc_id, 'Category': category, 'Location': location_en, 'Details': content_en})
                continue

            # TRACK A: LOCALIZED MODE
            matched_roads = self.road_df[self.road_df['STREET_ENAME'] == main_road]
            if matched_roads.empty:
                continue

            matched_roads = self.filter_directional_bounds(matched_roads, bound_compass)

            target_road_geom = matched_roads.geometry.unary_union
            pts_to_check = []

            sorted_roads = sorted(list(self.road_names_cache), key=len, reverse=True)
            for cross_road in sorted_roads:
                if cross_road == main_road: continue
                if cross_road in active_text_block or cross_road in location_en:
                    cross_feats = self.road_df[self.road_df['STREET_ENAME'] == cross_road]
                    if not cross_feats.empty:
                        cross_geom = cross_feats.geometry.unary_union
                        intersection = target_road_geom.intersection(cross_geom)
                        if not intersection.is_empty:
                            pts_to_check.append(intersection.centroid)
                        else:
                            p1, p2 = nearest_points(target_road_geom, cross_geom)
                            pts_to_check.append(p1)

            is_junction = "JUNCTION" in content_upper or "JUNCTION" in location_en
            if is_junction and pts_to_check:
                spatial_features_list.append({
                    'IncidentID': inc_id, 'RoadName': main_road, 
                    'RouteID': 'INTERSECTION_NODE', 'geometry': pts_to_check[0]
                })
                incident_records.append({'IncidentID': inc_id, 'Category': category, 'Location': location_en, 'Details': content_en})
                continue

            # Context Anchors
            landmark_en = self._get_xml_text(message, 'NEAR_LANDMARK_EN').upper().strip()
            between_en = self._get_xml_text(message, 'BETWEEN_LANDMARK_EN').upper().strip()
            
            if landmark_en in self.landmark_cache: pts_to_check.append(self.landmark_cache[landmark_en])
            if between_en in self.landmark_cache: pts_to_check.append(self.landmark_cache[between_en])

            is_between_incident = "BETWEEN" in content_upper or "BETWEEN" in location_en
            if is_between_incident and len(pts_to_check) >= 2:
                bounding_buffers = [MultiPoint(pts_to_check).envelope.buffer(40)]
            else:
                bounding_buffers = [pt.buffer(self.distance_threshold) for pt in pts_to_check]

            is_queueing = "TRAFFIC QUEUE" in content_upper
            spatial_match_count = 0
            
            for _, road_feat in matched_roads.iterrows():
                geom = road_feat['GEOMETRY']
                if geom is None: continue
                route_id_val = str(road_feat.get('ROUTE_ID', 'UNKNOWN'))

                def append_feat():
                    spatial_features_list.append({'IncidentID': inc_id, 'RoadName': main_road, 'RouteID': route_id_val, 'geometry': geom})

                if not bounding_buffers or is_queueing:
                    append_feat()
                    spatial_match_count += 1
                else:
                    if any(geom.intersects(buf) for buf in bounding_buffers):
                        append_feat()
                        spatial_match_count += 1
                        
            if spatial_match_count > 0:
                incident_records.append({'IncidentID': inc_id, 'Category': category, 'Location': location_en, 'Details': content_en})

        df_incidents = pd.DataFrame(incident_records).drop_duplicates(subset=['IncidentID'])
        if spatial_features_list:
            gdf_spatial = gpd.GeoDataFrame(spatial_features_list, crs=self.road_df.crs)
        else:
            gdf_spatial = gpd.GeoDataFrame(columns=['IncidentID', 'RoadName', 'RouteID', 'geometry'], crs=self.road_df.crs)
            
        active_ids = gdf_spatial['IncidentID'].unique()
        df_incidents = df_incidents[df_incidents['IncidentID'].isin(active_ids)].reset_index(drop=True)
        
        return df_incidents, gdf_spatial
