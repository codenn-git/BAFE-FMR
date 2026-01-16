# Incremental update system for FMR database
import os
import time
import pandas as pd
import geopandas as gpd
import rasterio
from datetime import datetime
import re
from shapely.geometry import box
from pyproj import Transformer
import numpy as np
import json

class IncrementalUpdater:
    """Handles incremental updates to FMR database without full rebuilds"""
    
    def __init__(self, shapefile_path, bsg_folder):
        self.shapefile_path = shapefile_path
        self.bsg_folder = bsg_folder
        self.db_path = os.path.join(os.path.dirname(shapefile_path), "fmr_database.csv")
        self.cache_file = os.path.join(os.path.dirname(shapefile_path), ".fmr_cache.json")
        self.load_cache()
        
    def load_cache(self):
        """Load cached information about processed files"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {'processed_rasters': {}, 'processed_shapefiles': {}, 'last_update': 0}
        else:
            self.cache = {'processed_rasters': {}, 'processed_shapefiles': {}, 'last_update': 0}
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def check_for_updates(self):
        """Quick check for new files without processing them"""
        status = {
            'new_shapefiles': [],
            'new_rasters': [],
            'has_updates': False
        }
        
        # Check for new shapefiles
        shapefile_dir = os.path.dirname(self.shapefile_path)
        for file in os.listdir(shapefile_dir):
            if file.endswith('.shp') and file != os.path.basename(self.shapefile_path):
                file_path = os.path.join(shapefile_dir, file)
                mtime = os.path.getmtime(file_path)
                
                if file not in self.cache['processed_shapefiles'] or \
                   self.cache['processed_shapefiles'][file] < mtime:
                    status['new_shapefiles'].append(file)
        
        # Check for new rasters
        if os.path.exists(self.bsg_folder):
            for file in os.listdir(self.bsg_folder):
                if file.endswith('Tiff.tif'):
                    file_path = os.path.join(self.bsg_folder, file)
                    mtime = os.path.getmtime(file_path)
                    
                    if file not in self.cache['processed_rasters'] or \
                       self.cache['processed_rasters'][file] < mtime:
                        status['new_rasters'].append(file)
        
        status['has_updates'] = bool(status['new_shapefiles'] or status['new_rasters'])
        return status
    
    def process_new_shapefiles(self, new_shapefiles):
        """Merge only new shapefiles into master"""
        if not new_shapefiles:
            return False
            
        print(f"Processing {len(new_shapefiles)} new shapefiles...")
        
        master_gdf = gpd.read_file(self.shapefile_path)
        master_crs = master_gdf.crs
        gdfs_to_merge = []
        
        shapefile_dir = os.path.dirname(self.shapefile_path)
        
        for shp_file in new_shapefiles:
            file_path = os.path.join(shapefile_dir, shp_file)
            try:
                gdf = gpd.read_file(file_path)
                if gdf.crs != master_crs:
                    gdf = gdf.to_crs(master_crs)
                gdfs_to_merge.append(gdf)
                
                # Update cache
                self.cache['processed_shapefiles'][shp_file] = os.path.getmtime(file_path)
                
                # Move to merged folder
                merged_folder = os.path.join(shapefile_dir, "merged")
                os.makedirs(merged_folder, exist_ok=True)
                
                base_name = os.path.splitext(shp_file)[0]
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = os.path.join(shapefile_dir, base_name + ext)
                    if os.path.exists(src):
                        dst = os.path.join(merged_folder, base_name + ext)
                        os.rename(src, dst)
                        
            except Exception as e:
                print(f"Error processing {shp_file}: {e}")
                continue
        
        if gdfs_to_merge:
            merged_gdf = pd.concat([master_gdf] + gdfs_to_merge, ignore_index=True)
            merged_gdf.to_file(self.shapefile_path)
            print(f"Merged {len(gdfs_to_merge)} new shapefiles into master")
            return True
        
        return False
    
    def process_new_rasters(self, new_rasters):
        """Process only new raster files and add to database"""
        if not new_rasters:
            return []
        
        print(f"Processing {len(new_rasters)} new raster files...")
        
        # Load FMRs
        fmr_gdf = gpd.read_file(self.shapefile_path).to_crs("EPSG:32651")
        raster_to_fmr_crs = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
        
        # Load existing database
        if os.path.exists(self.db_path):
            existing_df = pd.read_csv(self.db_path)
            existing_keys = set(
                zip(existing_df["FMR"], existing_df["BSG"], existing_df["Date"])
            )
        else:
            existing_df = pd.DataFrame()
            existing_keys = set()
        
        new_entries = []
        
        for tif_file in new_rasters:
            tif_path = os.path.join(self.bsg_folder, tif_file)
            
            try:
                with rasterio.open(tif_path) as src:
                    # Get raster bounds and transform to FMR CRS
                    minx, miny, maxx, maxy = src.bounds
                    minx_t, miny_t = raster_to_fmr_crs.transform(minx, miny)
                    maxx_t, maxy_t = raster_to_fmr_crs.transform(maxx, maxy)
                    raster_bounds = box(minx_t, miny_t, maxx_t, maxy_t)
                    
                    # Check which FMRs intersect with this raster
                    for idx, row in fmr_gdf.iterrows():
                        fmr_name = str(row.get("name", f"FMR-{idx}"))
                        fmr_geom = row.geometry
                        
                        if not fmr_geom.intersects(raster_bounds):
                            continue
                        
                        # Check for nodata coverage (as in original code)
                        geom_proj = gpd.GeoSeries([fmr_geom], crs=fmr_gdf.crs).to_crs(src.crs)
                        fmr_line = geom_proj.iloc[0]
                        
                        # Sample points along line
                        N = 10  # meters between sample points
                        num_segments = max(2, int(fmr_line.length / N))
                        sample_points = [
                            fmr_line.interpolate(dist) 
                            for dist in np.linspace(0, fmr_line.length, num_segments)
                        ]
                        coords = [(pt.x, pt.y) for pt in sample_points]
                        values = list(src.sample(coords))
                        
                        # Reject if any point lies on nodata
                        nodata_val = src.nodata if src.nodata is not None else 0
                        if any(val[0] == nodata_val or val[0] == 0 for val in values):
                            continue
                        
                        # Extract date/time from filename
                        match = re.search(r"(\d{8})-(\d{6})", tif_file)
                        if match:
                            raw_date, raw_time = match.groups()
                            try:
                                dt = datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")
                                formatted_date = dt.strftime("%Y-%m-%d")
                                formatted_time = dt.strftime("%H:%M:%S")
                            except ValueError:
                                formatted_date, formatted_time = "", ""
                        else:
                            formatted_date, formatted_time = "", ""
                        
                        # Skip if already in database
                        if (fmr_name, tif_file, formatted_date) in existing_keys:
                            continue
                        
                        new_entries.append({
                            "FMR": fmr_name,
                            "BSG": tif_file,
                            "Date": formatted_date,
                            "Time": formatted_time,
                            "Planned FMR Length": fmr_geom.length,
                            "Current FMR Length": "",
                            "FMR Progress": "",
                            "FMR Status": "",
                            "Mean FMR Width": "",
                            "Processing Type": "",
                            "Image Path": tif_path
                        })
                        
            except Exception as e:
                print(f"Error processing {tif_file}: {e}")
                continue
            
            # Update cache
            self.cache['processed_rasters'][tif_file] = os.path.getmtime(tif_path)
        
        # Append new entries to database
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            
            if not existing_df.empty:
                # Ensure columns match
                for col in existing_df.columns:
                    if col not in new_df.columns:
                        new_df[col] = ""
                
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df
            
            # Sort by FMR index and date
            final_df["FMR_INDEX"] = final_df["FMR"].str.extract(r"(\d+)", expand=False).fillna(0).astype(int)
            final_df["Date"] = pd.to_datetime(final_df["Date"], errors="coerce")
            final_df = final_df.sort_values(by=["FMR_INDEX", "Date"])
            final_df = final_df.drop(columns=["FMR_INDEX"])
            
            final_df.to_csv(self.db_path, index=False)
            print(f"Added {len(new_entries)} new entries to database")
        
        self.save_cache()
        return new_entries
    
    def perform_incremental_update(self):
        """Perform a complete incremental update"""
        status = self.check_for_updates()
        
        results = {
            'shapefiles_updated': False,
            'new_database_entries': 0,
            'status': 'success'
        }
        
        try:
            # Process new shapefiles
            if status['new_shapefiles']:
                results['shapefiles_updated'] = self.process_new_shapefiles(status['new_shapefiles'])
            
            # Process new rasters
            if status['new_rasters']:
                new_entries = self.process_new_rasters(status['new_rasters'])
                results['new_database_entries'] = len(new_entries)
            
            self.cache['last_update'] = time.time()
            self.save_cache()
            
            return results
            
        except Exception as e:
            print(f"Error during incremental update: {e}")
            return {
                'shapefiles_updated': False,
                'new_database_entries': 0,
                'status': 'error',
                'message': str(e)
            }
    
    def clear_cache(self):
        """Clear the cache to force full reprocessing"""
        self.cache = {'processed_rasters': {}, 'processed_shapefiles': {}, 'last_update': 0}
        self.save_cache()
        print("Cache cleared - next update will reprocess all files")
