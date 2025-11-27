"""
Refactored automatic processing for FMR GUI
Follows the clean pipeline structure from BSG_executable.py
Adds automatic image type detection
"""

import os
import cv2
import numpy as np
import geopandas as gpd
from datetime import datetime
from pathlib import Path
from shapely.geometry import shape

from utilv3 import (
    Preprocessing, Filters, Morph, MeasureWidth,
    measure_line, measure_line_transects, export
)

class ManualRoadProcessor:
    """
    Manual road processor that processes satellite images using user-drawn centerlines.
    """
    
    def __init__(self, manual_centerline_geojson, selected_fmr_gdf, raster_path,
                 output_base_dir, geojson_output_dir, image_type="", fmr_name=None):
        """
        Initialize the manual processor.
        
        Args:
            manual_centerline_geojson: GeoJSON dict of user-drawn centerline (WGS84)
            selected_fmr_gdf: GeoDataFrame with selected FMR (for reference/naming only)
            raster_path: Path to satellite image to process
            output_base_dir: Base directory for detailed outputs
            geojson_output_dir: Directory for consolidated GeoJSON outputs
            image_type: Image type (BSG, PNEO, SkySat) - auto-detected if not provided
        """
        self.manual_geojson = manual_centerline_geojson
        self.selected_fmr_gdf = selected_fmr_gdf
        self.raster_path = raster_path
        self.output_base_dir = output_base_dir
        self.geojson_output_dir = geojson_output_dir

        if self.selected_fmr_gdf.crs is None:
            self.selected_fmr_gdf = self.selected_fmr_gdf.set_crs('EPSG:4326')

        if fmr_name is not None:
            self.fmr_name = fmr_name
        else:
            self.fmr_name = self._extract_fmr_name()
        
        self.image_type = self._detect_image_type(raster_path)
        
        # Convert manual centerline to GeoDataFrame (this is the "planned FMR")
        self.manual_centerline_wgs84 = self._parse_manual_centerline()
        self.manual_centerline_metric = self.manual_centerline_wgs84.to_crs("EPSG:32651")
        
        # Create timestamped output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(
            output_base_dir,
            "Manual", 
            self.fmr_name,
            f"{self.fmr_name}_{timestamp}"
        )
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'FMR_ID': self.fmr_name,
            'TIMESTAMP': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_type': self.image_type,
            'processing_type': 'manual'
        }
        
        # Storage for processing products
        self.clipped_data = None
        self.clipped_transform = None
        self.final_binary_raster = None
        self.final_binary_transform = None
        self.transects = None
        self.detected_centerline = None  # Extracted from image
        self.road_polygon = None
    
    def _extract_fmr_name(self):
        """Extract FMR name from GeoDataFrame."""
        name_columns = ['name', 'FMR', 'FMR_ID', 'fmr_name', 'id', 'ID']
        
        if len(self.selected_fmr_gdf) > 0:
            for col in name_columns:
                if col in self.selected_fmr_gdf.columns:
                    name = self.selected_fmr_gdf.iloc[0].get(col)
                    if name and str(name) not in ['nan', 'None', '']:
                        return str(name)
        
        # Fallback: try to get index
        if len(self.selected_fmr_gdf) > 0:
            return f"FMR_{self.selected_fmr_gdf.index[0]}"
        
        return "Unknown_FMR"
    
    @staticmethod
    def _detect_image_type(raster_path):
        """Auto-detect image type from filename."""
        filename = Path(raster_path).name.lower()
        
        if 'bsg' in filename:
            return 'BSG'
        elif 'pneo' in filename:
            return 'PNEO'
        elif 'skysat' in filename:
            return 'SkySat'
        else:
            print(f"Warning: Could not detect image type from filename, defaulting to BSG")
            return 'BSG'
    
    def _parse_manual_centerline(self):
        """Parse manual centerline from GeoJSON to GeoDataFrame."""
        try:
            geom = shape(self.manual_geojson)
            gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:4326')
            return gdf
        except Exception as e:
            raise ValueError(f"Invalid manual centerline geometry: {str(e)}")
    
    def preprocess_image(self):
        """Reproject and clip the input raster using manual centerline."""
        print(f"[{self.fmr_name}] Preprocessing {self.image_type} image...")
        
        if self.image_type == 'BSG':
            preprocessor = Preprocessing()
            preprocessor.reproject(self.raster_path)

            # Clip with buffer for processing (manual centerline)
            self.clipped_data, self.clipped_transform = preprocessor.clipraster(
                vector_data=self.manual_centerline_metric,
                buffer_dist=25,
                bbox=False
            )

        elif self.image_type in ['PNEO', 'SkySat']:
            preprocessor = Preprocessing(pneo=True)
            preprocessor.reproject(self.raster_path)

            # Clip bounding-box for processing (manual centerline)
            self.clipped_data, self.clipped_transform = preprocessor.clipraster(
                vector_data=self.manual_centerline_metric,
                buffer_dist=25,
                bbox=True
            )

        # 11/27: export clipped original raster (common for all image types)
        clipped_path = os.path.join(self.output_folder, f"{self.fmr_name}_clipped.tif")
        export(
            self.clipped_data,
            clipped_path,
            'raster',
            'EPSG:32651',
            raster_transform=self.clipped_transform
        )

        return preprocessor
    
    def generate_binary_raster(self):
        """Apply image-specific filters and create binary road mask."""
        print(f"[{self.fmr_name}] Generating binary raster using {self.image_type} workflow...")
        
        filter_obj = Filters()
        morph = Morph()
        
        if self.image_type == 'BSG':
            # BSG processing: warmth enhancement + stretch
            warm_raster = filter_obj.enhance_image_warmth(self.clipped_data)
            stretch_raster = filter_obj.enhance_linear_stretch(self.clipped_data)
            
            morph_warm = morph.process(warm_raster, a=4, b=3, ite1=2, ite2=3)
            morph_stretch = morph.process(stretch_raster, a=4, b=3, ite1=2, ite2=3)
            
            binary_raster = np.logical_or(morph_warm, morph_stretch)
            
        elif self.image_type in ['PNEO', 'SkySat']:
            # PNEO/SkySat processing: CIELAB thresholding
            cielab = filter_obj.cielab(self.clipped_data)
            binary_raster = morph.threshold_cielab(cielab)
        
        else:
            raise ValueError(f"Unsupported image type: {self.image_type}")
        
        # Clean up small islands
        binary_raster = morph.remove_small_islands(binary_raster, min_size=500)
        binary_raster = np.squeeze(binary_raster)
        
        # Final morphological closing
        kernel = np.ones((3, 3), np.uint8)
        self.final_binary_raster = cv2.morphologyEx(
            binary_raster.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=3
        )
        
        self.final_binary_transform = self.clipped_transform
        
        # Export binary raster
        binary_path = os.path.join(self.output_folder, f"{self.fmr_name}_binary_raster.tif")
        export(self.final_binary_raster, binary_path, 'raster', 'EPSG:32651', 
               raster_transform=self.final_binary_transform)
    
    #11/21: robust manual width measurement (handles missing vectorized roads, exports via GeoPandas)
    def measure_width(self, interval=3, tolerance=0.15, resolution=0.3):
        """Generate transects and measure road width."""
        print(f"[{self.fmr_name}] Measuring road width (manual)...")

        if self.final_binary_raster is None or self.final_binary_transform is None:
            print(f"[{self.fmr_name}] Warning: Binary raster not available; skipping width measurement.")
            self.results['mean_width_m'] = None
            return None

        measure = MeasureWidth(
            self.final_binary_raster,
            self.final_binary_transform,
            self.manual_centerline_metric
        )

        # Run full MeasureWidth pipeline, but handle failures gracefully
        try:
            self.transects = measure.process(int=interval, tol=tolerance, res=resolution)
        except Exception as e:
            print(f"[{self.fmr_name}] Warning during width measurement: {e}")
            self.transects = None

        if self.transects is not None and not self.transects.empty:
            road_mean_width = self.transects['width'].mean()

            self.results['mean_width_m'] = float(road_mean_width)
            self.results['width_std'] = float(self.transects['width'].std())
            self.results['width_min'] = float(self.transects['width'].min())
            self.results['width_max'] = float(self.transects['width'].max())

            from scipy import stats
            # calculating Margin of Error
            n = len(self.transects['width'])
            t_critical = stats.t.ppf(1 - 0.05 / 2, df=n - 1)
            self.results['width_MoE'] = t_critical * (self.results['width_std'] / np.sqrt(n))

            # Export transects directly via GeoPandas (avoid utilv3.export CRS issues)
            transects_path = os.path.join(self.output_folder, f"{self.fmr_name}_transects.shp")
            try:
                self.transects.to_file(transects_path)
                print(f"[{self.fmr_name}] Transects exported to {transects_path}")
            except Exception as e:
                print(f"[{self.fmr_name}] Warning: Failed to export transects shapefile: {e}")

            return road_mean_width
        else:
            print(f"[{self.fmr_name}] Warning: No valid transects found (manual)")
            self.results['mean_width_m'] = None
            return None

    def create_centerline(self):
        if self.manual_centerline_metric is not None and not self.manual_centerline_metric.empty:

            manual_length = self.manual_centerline_metric.length.iloc[0]
            planned_fmr_metric = self.selected_fmr_gdf.to_crs("EPSG:32651")
            planned_length = planned_fmr_metric.length.iloc[0]

            progress_percent = (manual_length / planned_length) * 100 if planned_length > 0 else 0
            status = self._determine_status(progress_percent)

            self.results['manual_length_m'] = float(manual_length)
            self.results['planned_length_m'] = float(planned_length)
            self.results['progress_percent'] = float(progress_percent)
            self.results['status'] = status

            centerline_shp = os.path.join(self.output_folder, f"{self.fmr_name}_centerline.shp")
            export(self.manual_centerline_metric, centerline_shp, 'vector', 'EPSG:32651')

        else:
            print(f"[{self.fmr_name}] Warning: No centerline drawn")
            self.results['manual_length_m'] = float(self.manual_centerline_metric.length.iloc[0])
            self.results['progress_percent'] = 0
            self.results['status'] = "Not Started"
    
    #11/11 (migo): create polygon creation by buffering manual_centerline_metric using
    def generate_polygon(self, mean_road_width):
        """Create a road polygon by buffering the manual centerline using measured mean width."""
        if self.manual_centerline_metric is None or self.manual_centerline_metric.empty:
            print(f"[{self.fmr_name}] Warning: No manual centerline available for polygon generation")
            return None

        width = mean_road_width if mean_road_width is not None else self.results.get('mean_width_m')
        if width is None:
            print(f"[{self.fmr_name}] Warning: No mean width available to buffer centerline")
            return None

        try:
            buffer_dist = float(width) / 2.0

            gdf = self.manual_centerline_metric.copy()
            gdf['geometry'] = gdf.geometry.buffer(buffer_dist)

            # Save polygon and update results
            self.road_polygon = gdf
            self.results['mean_width_m'] = float(width)

            polygon_shp = os.path.join(self.output_folder, f"{self.fmr_name}_polygon.shp")
            export(self.road_polygon, polygon_shp, 'vector', 'EPSG:32651')

        except Exception as e:
            print(f"[{self.fmr_name}] Error generating polygon: {e}")
            return None
    
    def export_to_geojson(self):
        """Export detected centerline and polygon to consolidated GeoJSON files."""
        print(f"[{self.fmr_name}] Exporting to GeoJSON...")
        
        os.makedirs(self.geojson_output_dir, exist_ok=True)
        
        centerlines_path = os.path.join(self.geojson_output_dir, "fmr_centerlines_migo.geojson")
        polygons_path = os.path.join(self.geojson_output_dir, "fmr_polygons_migo.geojson")
        
        output_paths = {}
        
        # Export detected centerline
        #11/11 (migo): export drawn centerline
        if self.manual_centerline_metric is not None and not self.manual_centerline_metric.empty:
            centerline_wgs = self.manual_centerline_metric.copy().to_crs("EPSG:4326")
            
            # Add attributes
            centerline_wgs['FMR_ID'] = self.results['FMR_ID']
            centerline_wgs['TIMESTAMP'] = self.results['TIMESTAMP']
            centerline_wgs['length_m'] = self.results.get('manual_length_m', '')
            centerline_wgs['progress_percent'] = self.results.get('progress_percent', '')
            centerline_wgs['status'] = self.results.get('status', '')
            centerline_wgs['mean_width_m'] = self.results.get('mean_width_m', '')
            centerline_wgs['width_MoE'] = self.results.get('width_MoE', '')
            centerline_wgs['image_type'] = self.results['image_type']
            centerline_wgs['processing_type'] = 'manual'
            
            # Merge with existing centerlines
            centerlines_path = self._merge_to_geojson(
                centerline_wgs, 
                centerlines_path,
                dedupe_cols=['FMR_ID', 'processing_type']
            )
            output_paths['centerlines_geojson'] = centerlines_path
        
        # Export polygon if it exists
        if self.road_polygon is not None and not self.road_polygon.empty:
            polygon_wgs = self.road_polygon.copy().to_crs("EPSG:4326")
            
            # Add attributes
            polygon_wgs['FMR_ID'] = self.results['FMR_ID']
            polygon_wgs['TIMESTAMP'] = self.results['TIMESTAMP']
            polygon_wgs['mean_width_m'] = self.results.get('mean_width_m', '')
            polygon_wgs['width_MoE'] = self.results.get('width_MoE', '')
            polygon_wgs['image_type'] = self.results['image_type']
            polygon_wgs['processing_type'] = 'manual'
            
            # Merge with existing polygons
            polygons_path = self._merge_to_geojson(
                polygon_wgs,
                polygons_path,
                dedupe_cols=['FMR_ID', 'processing_type']
            )
            output_paths['polygons_geojson'] = polygons_path
        
        return output_paths
    
    @staticmethod
    def _merge_to_geojson(new_gdf, target_path, dedupe_cols):
        """Merge new features into existing GeoJSON, removing duplicates."""
        if os.path.exists(target_path):
            try:
                existing = gpd.read_file(target_path)
                
                # Remove duplicates based on dedupe_cols
                mask = existing[dedupe_cols[0]] == new_gdf.iloc[0][dedupe_cols[0]]
                for col in dedupe_cols[1:]:
                    mask &= existing[col] == new_gdf.iloc[0][col]
                
                existing = existing[~mask]
                
                # Combine
                merged = gpd.GeoDataFrame(
                    gpd.pd.concat([existing, new_gdf], ignore_index=True),
                    crs=new_gdf.crs
                )
            except Exception as e:
                print(f"Warning: Could not merge with existing GeoJSON: {e}")
                merged = new_gdf
        else:
            merged = new_gdf
        
        # Ensure timestamp is string
        if 'TIMESTAMP' in merged.columns:
            merged['TIMESTAMP'] = merged['TIMESTAMP'].astype(str)
        
        # Save
        merged.to_file(target_path, driver='GeoJSON')
        
        return target_path
    
    @staticmethod
    def _determine_status(progress_percent):
        """Determine project status based on progress percentage."""
        if progress_percent == 0:
            return "Not Started"
        elif progress_percent < 90:
            return "On-going"
        elif 90 <= progress_percent <= 110:
            return "Completed"
        else:
            return "Erroneous Processing"
    
    def run(self):
        """Execute the full manual processing pipeline with image processing."""
        print(f"\n{'='*60}")
        print(f"Manual Processing: {self.fmr_name}")
        print(f"Image Type: {self.image_type} (auto-detected)")
        print(f"Manual centerline will be used as reference")
        print(f"{'='*60}")
        
        try:
            # Step 1: Preprocess image
            self.preprocess_image()
            
            # Step 2: Generate binary raster
            self.generate_binary_raster()
            
            # Step 3: Measure width
            road_mean_width = self.measure_width()
            
            # Step 4: Extract centerline from image
            self.create_centerline()
            
            # Step 5: Generate polygon
            self.generate_polygon(road_mean_width)
            
            # Step 6: Export to GeoJSON
            output_paths = self.export_to_geojson()
            
            print(f"\n✓ Manual processing complete for {self.fmr_name}")
            print(f"  - Status: {self.results.get('status', 'Unknown')}")
            print(f"  - Manual length: {self.results.get('manual_length_m', 0):.1f}m")
            print(f"  - Progress: {self.results.get('progress_percent', 0):.1f}%")
            if road_mean_width:
                print(f"  - Mean Width: {road_mean_width:.2f}m")
            
            return {
                "status": "success",
                "results": self.results,
                "output_paths": output_paths,
                "output_folder": self.output_folder
            }
            
        except Exception as e:
            print(f"\n✗ Error processing {self.fmr_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": str(e),
                "results": self.results
            }

class AutomaticRoadProcessor:
    """
    Automatic road delineation processor with image type auto-detection.
    """
    
    def __init__(self, fmr_gdf, raster_path, output_base_dir, geojson_output_dir, fmr_name=None):
        """
        Initialize the automatic processor.
        
        Args:
            fmr_gdf: GeoDataFrame with single FMR geometry
            raster_path: Path to input raster image
            output_base_dir: Base directory for detailed outputs (timestamped folders)
            geojson_output_dir: Directory for consolidated GeoJSON outputs
        """
        self.fmr_gdf = fmr_gdf
        self.raster_path = raster_path
        self.output_base_dir = output_base_dir
        self.geojson_output_dir = geojson_output_dir
        
        if self.fmr_gdf.crs is None:
            self.fmr_gdf = self.fmr_gdf.set_crs('EPSG:32651')
        
        if fmr_name is not None:
            self.fmr_name = fmr_name
        else:
            self.fmr_name = self._extract_fmr_name()
        
        # Auto-detect image type
        self.image_type = self._detect_image_type(raster_path)

        # Create timestamped output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(
            output_base_dir,
            "Automatic", 
            self.fmr_name,
            f"{self.fmr_name}_{timestamp}"
        )
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'FMR_ID': self.fmr_name,
            'TIMESTAMP': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_type': self.image_type,
            'processing_type': 'automatic'
        }
        
        # Storage for processing products
        self.clipped_data = None
        self.clipped_transform = None
        self.final_binary_raster = None
        self.final_binary_transform = None
        self.transects = None
        self.centerline = None
        self.road_polygon = None
        
    def _extract_fmr_name(self):
        """Extract FMR name from GeoDataFrame."""
        name_columns = ['name', 'FMR', 'FMR_ID', 'fmr_name', 'id', 'ID']
        
        if len(self.fmr_gdf) > 0:
            for col in name_columns:
                if col in self.fmr_gdf.columns:
                    name = self.fmr_gdf.iloc[0].get(col)
                    if name and str(name) not in ['nan', 'None', '']:
                        return str(name)
        
        # Fallback: try to get index
        if len(self.fmr_gdf) > 0:
            return f"FMR_{self.fmr_gdf.index[0]}"
        
        return "Unknown_FMR"
    
    @staticmethod
    def _detect_image_type(raster_path):
        """
        Auto-detect image type from filename.
        
        Args:
            raster_path: Path to raster file
            
        Returns:
            str: 'BSG', 'PNEO', or 'SkySat'
        """
        filename = Path(raster_path).name.lower()
        
        if 'bsg' in filename:
            return 'BSG'
        elif 'pneo' in filename:
            return 'PNEO'
        elif 'skysat' in filename:
            return 'SkySat'
        else:
            # Default to BSG if unknown
            print(f"Warning: Could not detect image type from filename, defaulting to BSG")
            return 'BSG'
    
    def preprocess_image(self):
        """Reproject and clip the input raster."""
        print(f"[{self.fmr_name}] Preprocessing {self.image_type} image...")
        
        if self.image_type == 'BSG':
            preprocessor = Preprocessing()
            preprocessor.reproject(self.raster_path)
            
            # Clip with buffer for processing
            self.clipped_data, self.clipped_transform = preprocessor.clipraster(
                vector_data=self.fmr_gdf,
                buffer_dist=25,
                bbox=True
            )

        elif self.image_type in ['PNEO', 'SkySat']:
            preprocessor = Preprocessing(pneo=True)
            preprocessor.reproject(self.raster_path)

            # Clip bounding-box for processing
            self.clipped_data, self.clipped_transform = preprocessor.clipraster(
                vector_data=self.fmr_gdf,
                buffer_dist=25,
                bbox=True
            )

        # 11/27: export clipped original raster (common for all image types)
        clipped_path = os.path.join(self.output_folder, f"{self.fmr_name}_clipped.tif")
        export(
            self.clipped_data,
            clipped_path,
            'raster',
            'EPSG:32651',
            raster_transform=self.clipped_transform
        )

        return preprocessor
    
    def generate_binary_raster(self):
        """Apply image-specific filters and create binary road mask."""
        print(f"[{self.fmr_name}] Generating binary raster using {self.image_type} workflow...")
        
        filter_obj = Filters()
        morph = Morph()
        
        if self.image_type == 'BSG':
            warm_raster = filter_obj.enhance_image_warmth(self.clipped_data)
            stretch_raster = filter_obj.enhance_linear_stretch(self.clipped_data)
            
            morph_warm = morph.process(warm_raster, a=4, b=3, ite1=2, ite2=3)
            morph_stretch = morph.process(stretch_raster, a=4, b=3, ite1=2, ite2=3)
            
            binary_raster = np.logical_or(morph_warm, morph_stretch)
            
        elif self.image_type in ['PNEO', 'SkySat']:
            # PNEO/SkySat processing: CIELAB thresholding
            cielab = filter_obj.cielab(self.clipped_data)
            binary_raster = morph.threshold_cielab(cielab)
        
        else:
            raise ValueError(f"Unsupported image type: {self.image_type}")
        
        # Clean up small islands
        binary_raster = morph.remove_small_islands(binary_raster, min_size=500)
        binary_raster = np.squeeze(binary_raster)
        
        # Final morphological closing
        kernel = np.ones((3, 3), np.uint8)
        self.final_binary_raster = cv2.morphologyEx(
            binary_raster.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=3
        )
        
        self.final_binary_transform = self.clipped_transform
        
        # Export binary raster
        binary_path = os.path.join(self.output_folder, f"{self.fmr_name}_binary_raster.tif")
        export(self.final_binary_raster, binary_path, 'raster', 'EPSG:32651', 
               raster_transform=self.final_binary_transform)
        
    
    #11/21: robust automatic width measurement (handles missing vectorized roads, exports via GeoPandas)
    def measure_width(self, interval=3, tolerance=0.15, resolution=0.3):
        """Generate transects and measure road width."""
        print(f"[{self.fmr_name}] Measuring road width (automatic)...")

        if self.final_binary_raster is None or self.final_binary_transform is None:
            print(f"[{self.fmr_name}] Warning: Binary raster not available; skipping width measurement.")
            self.results['mean_width_m'] = None
            return None

        measure = MeasureWidth(
            self.final_binary_raster,
            self.final_binary_transform,
            self.fmr_gdf
        )

        # Run full MeasureWidth pipeline, but handle failures gracefully
        try:
            self.transects = measure.process(int=interval, tol=tolerance, res=resolution)
        except Exception as e:
            print(f"[{self.fmr_name}] Warning during width measurement: {e}")
            self.transects = None

        if self.transects is not None and not self.transects.empty:
            road_mean_width = self.transects['width'].mean()

            self.results['mean_width_m'] = float(road_mean_width)
            self.results['width_std'] = float(self.transects['width'].std())
            self.results['width_min'] = float(self.transects['width'].min())
            self.results['width_max'] = float(self.transects['width'].max())

            from scipy import stats
            # calculating Margin of Error
            n = len(self.transects['width'])
            t_critical = stats.t.ppf(1 - 0.05 / 2, df=n - 1)
            self.results['width_MoE'] = t_critical * (self.results['width_std'] / np.sqrt(n))

            # Export transects directly via GeoPandas (avoid utilv3.export CRS issues)
            transects_path = os.path.join(self.output_folder, f"{self.fmr_name}_transects.shp")
            try:
                self.transects.to_file(transects_path)
                print(f"[{self.fmr_name}] Transects exported to {transects_path}")
            except Exception as e:
                print(f"[{self.fmr_name}] Warning: Failed to export transects shapefile: {e}")

            print(f"[{self.fmr_name}] Mean width: {road_mean_width:.2f}m")
            return road_mean_width
        else:
            print(f"[{self.fmr_name}] Warning: No valid transects found (automatic)")
            self.results['mean_width_m'] = None
            return None

    def extract_centerline(self, preprocessor, spacing=5):
        """
        Extract road centerline with automatic fallback.
        Tries skeleton method first, falls back to transects if result is erroneous.
        
        Parameters:
        - preprocessor: Preprocessing object
        - spacing: spacing for skeleton method
        """
        print(f"[{self.fmr_name}] Extracting centerline...")
        
        # Clip binary raster tightly for centerline extraction
        binary_clipped, binary_transform = preprocessor.clipraster(
            raster_data=self.final_binary_raster.astype(np.uint8),
            vector_data=self.fmr_gdf,
            transform=self.clipped_transform,
            buffer_dist=3
        )
        binary_clipped = np.squeeze(binary_clipped)
        
        # Try skeleton-based method first
        # print(f"[{self.fmr_name}] Trying skeleton-based method...")
        centerline_skeleton = measure_line(
            binary_clipped,
            binary_transform,
            spacing=spacing,
            crs="EPSG:32651"
        )
        
        # Calculate progress for skeleton method
        actual_length_skeleton = centerline_skeleton.length.values[0]
        planned_length = self.fmr_gdf.to_crs("EPSG:32651").length.sum()
        progress_skeleton = (actual_length_skeleton / planned_length) * 100
        status_skeleton = self._determine_status(progress_skeleton)
        
        # print(f"[{self.fmr_name}] Skeleton method: {progress_skeleton:.1f}% ({status_skeleton})")
        
        # Check if skeleton method is erroneous
        if status_skeleton == "Erroneous Processing":
            # print(f"[{self.fmr_name}] Skeleton method erroneous, falling back to transect-based method...")
            
            # Ensure transects are available
            if self.transects is None:
                raise ValueError("Transects must be generated first. Run measure_width() before extract_centerline().")
            
            # Use transect-based method
            self.centerline = measure_line_transects(
                self.transects,
                crs="EPSG:32651",
                smooth=True
            )
            
            # Recalculate metrics with transect method
            actual_length = self.centerline.length.values[0]
            progress_percent = (actual_length / planned_length) * 100
            status = self._determine_status(progress_percent)
            
            print(f"[{self.fmr_name}] Transect method: {progress_percent:.1f}% ({status})")
            
            self.results['method'] = 'transects'
            self.results['fallback_reason'] = 'skeleton_erroneous'
            self.results['skeleton_progress_percent'] = float(progress_skeleton)
        else:
            # Skeleton method is good, use it
            # print(f"[{self.fmr_name}] Skeleton method successful!")
            self.centerline = centerline_skeleton
            actual_length = actual_length_skeleton
            progress_percent = progress_skeleton
            status = status_skeleton
            
            self.results['method'] = 'skeleton'
        
        # Store final metrics
        self.results['length_m'] = float(actual_length)
        self.results['planned_length_m'] = float(planned_length)
        self.results['progress_percent'] = float(progress_percent)
        self.results['status'] = status
        
        # Export centerline
        centerline_shp = os.path.join(self.output_folder, f"{self.fmr_name}_centerline.shp")
        export(self.centerline, centerline_shp, 'vector', 'EPSG:32651')
        
        print(f"[{self.fmr_name}] Centerline extracted: {actual_length:.1f}m ({progress_percent:.1f}% complete)")
    
    def generate_polygon(self, road_mean_width):
        """
        Generate road polygon from the extracted centerline and mean width.
        Uses the centerline that was already extracted (either skeleton or transect-based).
        
        Parameters:
        - road_mean_width: mean width from transects
        - preprocessor: Preprocessing object (kept for compatibility)
        """
        if road_mean_width is None or self.centerline is None or self.centerline.empty:
            print(f"[{self.fmr_name}] Skipping polygon generation (no width or centerline)")
            return
        
        print(f"[{self.fmr_name}] Generating road polygon...")
        
        # Generate polygon by buffering the existing centerline
        centerline_geom = self.centerline.geometry.iloc[0]
        road_polygon_geom = centerline_geom.buffer(
            road_mean_width / 2, 
            cap_style=1,  # flat cap
            join_style=1   # round join
        )
        
        self.road_polygon = gpd.GeoDataFrame(
            geometry=[road_polygon_geom], 
            crs=self.centerline.crs
        )
        
        # Export polygon
        polygon_shp = os.path.join(self.output_folder, f"{self.fmr_name}_polygon.shp")
        export(self.road_polygon, polygon_shp, 'vector', 'EPSG:32651')
        
        # method = self.results.get('method', 'skeleton')
        # print(f"[{self.fmr_name}] Road polygon generated (from {method} centerline)")
    
    def export_to_geojson(self):
        """
        Export centerline and polygon to consolidated GeoJSON files.
        Follows the same structure as manual workflow.
        """
        print(f"[{self.fmr_name}] Exporting to GeoJSON...")
        
        os.makedirs(self.geojson_output_dir, exist_ok=True)
        
        centerlines_path = os.path.join(self.geojson_output_dir, "fmr_centerlines_migo.geojson")
        polygons_path = os.path.join(self.geojson_output_dir, "fmr_polygons_migo.geojson")
        
        output_paths = {}
        
        # Export centerline
        if self.centerline is not None and not self.centerline.empty:
            centerline_wgs = self.centerline.copy().to_crs("EPSG:4326")
            
            # Add attributes
            centerline_wgs['FMR_ID'] = self.results['FMR_ID']
            centerline_wgs['TIMESTAMP'] = self.results['TIMESTAMP']
            centerline_wgs['length_m'] = self.results.get('length_m', '')
            centerline_wgs['progress_percent'] = self.results.get('progress_percent', '')
            centerline_wgs['status'] = self.results.get('status', '')
            centerline_wgs['mean_width_m'] = self.results.get('mean_width_m', '')
            centerline_wgs['width_MoE'] = self.results.get('width_MoE', '')
            centerline_wgs['image_type'] = self.results['image_type']
            centerline_wgs['processing_type'] = 'automatic'
            centerline_wgs['method'] = self.results.get('method', 'skeleton')
            
            # Add fallback information if applicable
            if 'fallback_reason' in self.results:
                centerline_wgs['fallback_reason'] = self.results['fallback_reason']
                centerline_wgs['skeleton_progress'] = self.results.get('skeleton_progress_percent', '')
            
            # Merge with existing centerlines
            centerlines_path = self._merge_to_geojson(
                centerline_wgs, 
                centerlines_path,
                dedupe_cols=['FMR_ID', 'processing_type']
            )
            output_paths['centerlines_geojson'] = centerlines_path
        
        # Export polygon
        if self.road_polygon is not None and not self.road_polygon.empty:
            polygon_wgs = self.road_polygon.copy().to_crs("EPSG:4326")
            
            # Add attributes
            polygon_wgs['FMR_ID'] = self.results['FMR_ID']
            polygon_wgs['TIMESTAMP'] = self.results['TIMESTAMP']
            polygon_wgs['mean_width_m'] = self.results.get('mean_width_m', '')
            polygon_wgs['width_MoE'] = self.results.get('width_MoE', '')
            polygon_wgs['image_type'] = self.results['image_type']
            polygon_wgs['processing_type'] = 'automatic'
            polygon_wgs['method'] = self.results.get('method', 'skeleton')
            
            # Merge with existing polygons
            polygons_path = self._merge_to_geojson(
                polygon_wgs,
                polygons_path,
                dedupe_cols=['FMR_ID', 'processing_type']
            )
            output_paths['polygons_geojson'] = polygons_path
        
        return output_paths
    
    @staticmethod
    def _merge_to_geojson(new_gdf, target_path, dedupe_cols):
        """
        Merge new features into existing GeoJSON, removing duplicates.
        
        Args:
            new_gdf: GeoDataFrame with new features
            target_path: Path to target GeoJSON file
            dedupe_cols: List of columns to use for deduplication
        """
        if os.path.exists(target_path):
            try:
                existing = gpd.read_file(target_path)
                
                # Remove duplicates based on dedupe_cols
                mask = existing[dedupe_cols[0]] == new_gdf.iloc[0][dedupe_cols[0]]
                for col in dedupe_cols[1:]:
                    mask &= existing[col] == new_gdf.iloc[0][col]
                
                existing = existing[~mask]
                
                # Combine
                merged = gpd.GeoDataFrame(
                    gpd.pd.concat([existing, new_gdf], ignore_index=True),
                    crs=new_gdf.crs
                )
            except Exception as e:
                print(f"Warning: Could not merge with existing GeoJSON: {e}")
                merged = new_gdf
        else:
            merged = new_gdf
        
        # Ensure timestamp is string
        if 'TIMESTAMP' in merged.columns:
            merged['TIMESTAMP'] = merged['TIMESTAMP'].astype(str)
        
        # Save
        merged.to_file(target_path, driver='GeoJSON')
        
        return target_path
    
    @staticmethod
    def _determine_status(progress_percent):
        """Determine project status based on progress percentage."""
        if 0 <= progress_percent < 10:
            return "Not Started"
        elif 10 <= progress_percent < 90:
            return "On-going"
        elif 90 <= progress_percent <= 110:
            return "Completed"
        else:
            return "Erroneous Processing"
    
    def run(self):
        """Execute the full automatic processing pipeline."""
        print(f"\n{'='*60}")
        print(f"Automatic Processing: {self.fmr_name}")
        print(f"Image Type: {self.image_type} (auto-detected)")
        print(f"{'='*60}")
        
        try:
            # Step 1: Preprocess
            preprocessor = self.preprocess_image()
            
            # Step 2: Generate binary raster
            self.generate_binary_raster()
            
            # Step 3: Measure width
            road_mean_width = self.measure_width()
            
            # Step 4: Extract centerline
            self.extract_centerline(preprocessor)
            
            # Step 5: Generate polygon
            self.generate_polygon(road_mean_width)
            
            # Step 6: Export to GeoJSON
            output_paths = self.export_to_geojson()
            
            print(f"\n✓ Processing complete for {self.fmr_name}")
            print(f"  - Method: {self.results.get('method', 'skeleton')}")
            print(f"  - Status: {self.results.get('status', 'Unknown')}")
            print(f"  - Progress: {self.results.get('progress_percent', 0):.1f}%")
            if road_mean_width:
                print(f"  - Mean Width: {road_mean_width:.2f}m")
            
            if 'fallback_reason' in self.results:
                print(f"  - Fallback reason: {self.results['fallback_reason']}")
            
            return {
                "status": "success",
                "results": self.results,
                "output_paths": output_paths,
                "output_folder": self.output_folder
            }
            
        except Exception as e:
            print(f"\n✗ Error processing {self.fmr_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": str(e),
                "results": self.results
            }

def process_automatic(fmr_gdf, raster_path, output_base_dir, geojson_output_dir, fmr_name=None):
    """
    Wrapper function for automatic processing.
    Drop-in replacement for the old processing() function.
    
    Args:
        fmr_gdf: GeoDataFrame with FMR geometry
        raster_path: Path to raster image
        output_base_dir: Base directory for outputs
        geojson_output_dir: Directory for GeoJSON files
    
    Returns:
        dict: Processing results
    """
    processor = AutomaticRoadProcessor(
        fmr_gdf=fmr_gdf,
        raster_path=raster_path,
        output_base_dir=output_base_dir,
        geojson_output_dir=geojson_output_dir,
        fmr_name=fmr_name    
    )
    
    return processor.run()

def process_manual(manual_centerline_geojson, selected_fmr_gdf, raster_path,
                   output_base_dir, geojson_output_dir, image_type="", fmr_name=None):
    """
    Wrapper function for manual processing WITH image processing.
    Drop-in replacement for manual workflow in fmr_gui-new.py.
    
    Args:
        manual_centerline_geojson: GeoJSON dict of user-drawn centerline (WGS84)
        selected_fmr_gdf: GeoDataFrame with selected FMR (for reference/naming)
        raster_path: Path to satellite image to process
        output_base_dir: Base directory for outputs
        geojson_output_dir: Directory for GeoJSON files
        image_type: Optional image type label (auto-detected if not provided)
    
    Returns:
        dict: Processing results including detected centerline, width measurements, etc.
    """
    processor = ManualRoadProcessor(
        manual_centerline_geojson=manual_centerline_geojson,
        selected_fmr_gdf=selected_fmr_gdf,
        raster_path=raster_path,
        output_base_dir=output_base_dir,
        geojson_output_dir=geojson_output_dir,
        image_type=image_type,
        fmr_name=fmr_name
    )
    
    return processor.run()
