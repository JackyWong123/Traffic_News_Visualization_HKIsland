# traffic_processor.py
import xml.etree.ElementTree as ET
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint
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
        """Categorizes traffic reports based on text pattern matching."""
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

    def calculate_traffic_flow_heading(self, geom):
        """Calculates true traffic heading from the first coordinate to the last coordinate (Strictly Code 3)."""
        if geom is None:
            return None
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString' and not geom.is_empty:
            coords = list(max(geom.geoms, key=lambda l: l.length).coords)
        else:
            return None
            
        if len(coords) >= 2:
            # Target your logic: Calculate vector direction between first pair [0] and last pair [-1]
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            
            # Match the dominant heading direction axis
            if abs(dx) > abs(dy):
                return "EAST" if dx > 0 else "WEST"
            else:
                return "NORTH" if dy > 0 else "SOUTH"
        return None

    @staticmethod
    def _get_xml_text(message_element, tag_name):
        for child in message_element:
            if child.tag.split('}')[-1] == tag_name:
                return child.text.strip() if child.text else ""
        return ""

    def process_active_incidents(self):
        """Parses XML logs and handles explicit 1 vs 3 directional criteria."""
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
            landmark_en = self._get_xml_text(message, 'NEAR_LANDMARK_EN').upper().strip()
            between_en = self._get_xml_text(message, 'BETWEEN_LANDMARK_EN').upper().strip()
            content_en = self._get_xml_text(message, 'CONTENT_EN').strip()
            content_upper = content_en.upper()

            if status == "CLOSED" or not location_en: 
                continue
                
            active_text_block = content_upper.split("RESUMED NORMAL")[0]
            category = self.classify_incident(content_upper)

            # Translate news destinations into target tracking compass bounds
            bound_target = None
            if any(kw in content_upper or kw in location_en for kw in ["CENTRAL BOUND", "CENTRAL-BOUND", "KENNEDY TOWN BOUND"]):
                bound_target = "WEST"
            elif any(kw in content_upper or kw in location_en for kw in ["WAN CHAI BOUND", "CHAI WAN BOUND", "EAST BOUND", "CAUSEWAY BAY BOUND"]):
                bound_target = "EAST"
            elif any(kw in content_upper or kw in location_en for kw in ["ABERDEEN BOUND", "SOUTH BOUND", "WONG CHUK HANG BOUND"]):
                bound_target = "SOUTH"
            elif any(kw in content_upper or kw in location_en for kw in ["MONG KOK BOUND", "KOWLOON BOUND", "NORTH BOUND"]):
                bound_target = "NORTH"

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
                        
                        dir_col = next((c for c in matched_roads.columns if c in ['TRAVEL_DIRECTION', 'TRAFFIC_DIRECTION', 'DIR_CODE', 'DIRECTION']), None)
                        valid_indices = []
                        for idx, road_feat in matched_roads.iterrows():
                            geom = road_feat['GEOMETRY']
                            if geom is None: continue
                            
                            dir_val = "1"
                            if dir_col and pd.notna(road_feat[dir_col]):
                                dir_val = str(road_feat[dir_col]).strip().split('.')[0]
                                
                            # Value 1 = Two-Way. Keep it immediately.
                            if not bound_target or dir_val == '1':
                                valid_indices.append(idx)
                                continue
                                
                            # Value 3 = One-Way. Run your coordinate check.
                            if dir_val == '3':
                                seg_heading = self.calculate_traffic_flow_heading(geom)
                                if seg_heading == bound_target:
                                    valid_indices.append(idx)
                                    
                        if valid_indices:
                            matched_roads = matched_roads.loc[valid_indices]
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

            dir_col = next((c for c in matched_roads.columns if c in ['TRAVEL_DIRECTION', 'TRAFFIC_DIRECTION', 'DIR_CODE', 'DIRECTION']), None)

            # ST_TRACKER CARRIAGEWAY VALIDATION FILTER LOOP
            valid_indices = []
            for idx, road_feat in matched_roads.iterrows():
                geom = road_feat['GEOMETRY']
                if geom is None: continue
                
                if not bound_target:
                    valid_indices.append(idx)
                    continue
                
                # Float-safe casting cleanup (handles 1.0 or 3.0 flawlessly)
                dir_val = "1"
                if dir_col and pd.notna(road_feat[dir_col]):
                    dir_val = str(road_feat[dir_col]).strip().split('.')[0]
                
                # Rule 1: Code 1 means permitted in both directions. Keep it.
                if dir_val == '1':
                    valid_indices.append(idx)
                    continue
                    
                # Rule 2: Code 3 means permitted in digitized coordinates sequence direction only.
                if dir_val == '3':
                    seg_heading = self.calculate_traffic_flow_heading(geom)
                    if seg_heading == bound_target:
                        valid_indices.append(idx)
                        
            if valid_indices:
                matched_roads = matched_roads.loc[valid_indices]

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
