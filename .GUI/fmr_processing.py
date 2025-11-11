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

from utilv3 import (
    Preprocessing, Filters, Morph, MeasureWidth,
    measure_line, export
)


class AutomaticRoadProcessor:
    """
    Automatic road delineation processor with image type auto-detection.
    """
    
    def __init__(self, fmr_gdf, raster_path, fmr_name, output_base_dir, geojson_output_dir):
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
                bbox=False
            )
        
        elif self.image_type in ['PNEO', 'SkySat']:
            preprocessor = Preprocessing(pneo=True)
            preprocessor.reproject(self.raster_path)

            #Clip bounding-box
            self.clipped_data, self.clipped_transform = preprocessor.clipraster(
                vector_data=self.fmr_gdf,
                buffer_dist = 25,
                bbox=True
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
        
        print(f"[{self.fmr_name}] Binary raster saved: {binary_path}")
    
    def measure_width(self, interval=3, tolerance=0.15, resolution=0.3):
        """Generate transects and measure road width."""
        print(f"[{self.fmr_name}] Measuring road width...")
        
        measure = MeasureWidth(
            self.final_binary_raster,
            self.final_binary_transform,
            self.fmr_gdf
        )
        
        self.transects = measure.process(int=interval, tol=tolerance, res=resolution)
        
        if self.transects is not None and not self.transects.empty:
            road_mean_width = self.transects['width'].mean()
            
            self.results['mean_width_m'] = float(road_mean_width)
            self.results['width_std'] = float(self.transects['width'].std())
            self.results['width_min'] = float(self.transects['width'].min())
            self.results['width_max'] = float(self.transects['width'].max())
            
            from scipy import stats
            #calculating Margin of Error
            n = len(self.transects['width'])
            t_critical = stats.t.ppf(1 - 0.05 / 2, df = n - 1)
            self.results['width_MoE'] = t_critical * (self.results['width_std']/np.sqrt(n))
            
            # Export transects
            transects_path = os.path.join(self.output_folder, f"{self.fmr_name}_transects.shp")
            measure.export(transects_path, gdf=self.transects)
            
            print(f"[{self.fmr_name}] Mean width: {road_mean_width:.2f}m")
            return road_mean_width
        else:
            print(f"[{self.fmr_name}] Warning: No valid transects found")
            self.results['mean_width_m'] = None
            return None
    
    def extract_centerline(self, preprocessor, spacing=3):
        """Extract road centerline from binary raster."""
        print(f"[{self.fmr_name}] Extracting centerline...")
        
        # Clip binary raster tightly for centerline extraction
        binary_clipped, binary_transform = preprocessor.clipraster(
            raster_data=self.final_binary_raster.astype(np.uint8),
            vector_data=self.fmr_gdf,
            transform=self.clipped_transform,
            buffer_dist=3
        )
        binary_clipped = np.squeeze(binary_clipped)
        
        # Generate centerline
        self.centerline = measure_line(
            binary_clipped,
            binary_transform,
            spacing=spacing,
            crs="EPSG:32651"
        )
        
        if self.centerline is not None and not self.centerline.empty:
            # Calculate metrics
            actual_length = self.centerline.length.values[0]
            planned_length = self.fmr_gdf.to_crs("EPSG:32651").length.sum()
            
            progress_percent = (actual_length / planned_length) * 100 if planned_length > 0 else 0
            status = self._determine_status(progress_percent)
            
            self.results['length_m'] = float(actual_length)
            self.results['planned_length_m'] = float(planned_length)
            self.results['progress_percent'] = float(progress_percent)
            self.results['status'] = status
            
            # Export centerline
            centerline_shp = os.path.join(self.output_folder, f"{self.fmr_name}_centerline.shp")
            export(self.centerline, centerline_shp, 'vector', 'EPSG:32651')
            
            print(f"[{self.fmr_name}] Centerline extracted: {actual_length:.1f}m ({progress_percent:.1f}% complete)")
        else:
            print(f"[{self.fmr_name}] Warning: No centerline detected")
            self.results['length_m'] = None
            self.results['progress_percent'] = 0
            self.results['status'] = "Not Started"
    
    def generate_polygon(self, road_mean_width, preprocessor):
        """Generate road polygon using measured width."""
        if road_mean_width is None or self.centerline is None or self.centerline.empty:
            print(f"[{self.fmr_name}] Skipping polygon generation (no width or centerline)")
            return
        
        print(f"[{self.fmr_name}] Generating road polygon...")
        
        # Use the preprocessor that was already created (has CRS set)
        binary_clipped, binary_transform = preprocessor.clipraster(
            raster_data=self.final_binary_raster.astype(np.uint8),
            vector_data=self.fmr_gdf,
            transform=self.clipped_transform,
            buffer_dist=3
        )
        binary_clipped = np.squeeze(binary_clipped)
        
        self.road_polygon = measure_line(
            binary_clipped,
            binary_transform,
            road_width=road_mean_width,
            return_polygon=True,
            crs="EPSG:32651"
        )
        
        if self.road_polygon is not None and not self.road_polygon.empty:
            # Export polygon
            polygon_shp = os.path.join(self.output_folder, f"{self.fmr_name}_polygon.shp")
            export(self.road_polygon, polygon_shp, 'vector', 'EPSG:32651')
            
            print(f"[{self.fmr_name}] Road polygon generated")
    
    def export_to_geojson(self):
        """
        Export centerline and polygon to consolidated GeoJSON files.
        Follows the same structure as manual workflow.
        """
        print(f"[{self.fmr_name}] Exporting to GeoJSON...")
        
        os.makedirs(self.geojson_output_dir, exist_ok=True)
        
        centerlines_path = os.path.join(self.geojson_output_dir, "fmr_centerlines_aina.geojson")
        polygons_path = os.path.join(self.geojson_output_dir, "fmr_polygons.geojson")
        
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
        if progress_percent == 0:
            return "Not Started"
        elif progress_percent < 90:
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
            self.generate_polygon(road_mean_width, preprocessor)
            
            # Step 6: Export to GeoJSON
            output_paths = self.export_to_geojson()
            
            print(f"\n✓ Processing complete for {self.fmr_name}")
            print(f"  - Status: {self.results.get('status', 'Unknown')}")
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

def process_automatic(fmr_gdf, raster_path, fmr_name, output_base_dir, geojson_output_dir):
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
        fmr_name=fmr_name,
        output_base_dir=output_base_dir,
        geojson_output_dir=geojson_output_dir
    )
    
    return processor.run()

# class ManualRoadProcessor:
#     def __init__(self, fmr_gdf, raster_path, output_base_dir, geojson_output_dir):
#         """
#         Initialize the automatic processor.
        
#         Args:
#             fmr_gdf: GeoDataFrame with single FMR geometry
#             raster_path: Path to input raster image
#             output_base_dir: Base directory for detailed outputs (timestamped folders)
#             geojson_output_dir: Directory for consolidated GeoJSON outputs
#         """
#         self.fmr_gdf = fmr_gdf
#         self.raster_path = raster_path
#         self.output_base_dir = output_base_dir
#         self.geojson_output_dir = geojson_output_dir
        
#         if self.fmr_gdf.crs is None:
#             self.fmr_gdf = self.fmr_gdf.set_crs('EPSG:32651')
        
#         self.fmr_name = self._extract_fmr_name()
        
#         # Auto-detect image type
#         self.image_type = self._detect_image_type(raster_path)
        
#         # Create timestamped output folder
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.output_folder = os.path.join(
#             output_base_dir, 
#             f"{self.fmr_name}_{timestamp}"
#         )
#         os.makedirs(self.output_folder, exist_ok=True)
        
#         # Initialize results dictionary
#         self.results = {
#             'FMR_ID': self.fmr_name,
#             'TIMESTAMP': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'image_type': self.image_type,
#             'processing_type': 'automatic'
#         }
        
#         # Storage for processing products
#         self.clipped_data = None
#         self.clipped_transform = None
#         self.final_binary_raster = None
#         self.final_binary_transform = None
#         self.transects = None
#         self.centerline = None
#         self.road_polygon = None

# def manual_processing():