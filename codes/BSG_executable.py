"""
BlackSky Global (BSG) Road Delineation Pipeline
Processes BSG satellite imagery to extract road centerlines and polygons.
Outputs results to GeoJSON for web visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from utilv3 import (
    Preprocessing, Filters, Morph, MeasureWidth,
    measure_line, export
)
from rasterio.plot import show


class RoadDelineationPipeline:
    """Main pipeline for road delineation from satellite imagery."""
    
    def __init__(self, raster_path, vector_path, output_base_dir):
        """
        Initialize the pipeline.
        
        Args:
            raster_path: Path to input raster image
            vector_path: Path to input vector (planned FMR)
            output_base_dir: Base directory for outputs
        """
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.output_base_dir = output_base_dir
        
        # Extract vector basename for file naming
        self.vector_basename = Path(vector_path).stem
        
        # Create output directories
        self.output_folder = os.path.join(output_base_dir, self.vector_basename)
        self.image_output_path = os.path.join(self.output_folder, "images")
        os.makedirs(self.image_output_path, exist_ok=True)
        
        # Load vector data
        self.vector_gdf = gpd.read_file(vector_path)
        
        # Initialize results dictionary
        self.results = {
            'fmr_id': self.vector_basename,
            'timestamp': datetime.now().isoformat(),
            'image_type': 'BSG'
        }
        
        # Storage for intermediate products
        self.clipped_data = None
        self.clipped_transform = None
        self.final_binary_raster = None
        self.final_binary_transform = None
        self.transects = None
        self.centerline = None
        self.road_polygon = None
        self.display_image = None
        self.display_transform = None
        self.binary_clipped = None
        self.binary_transform = None
        
    @staticmethod
    def normalize_raster(raster):
        """Normalize raster values to 0-255 for display."""
        raster_min = np.nanmin(raster)
        raster_max = np.nanmax(raster)
        
        if raster_max == raster_min:
            return np.zeros_like(raster, dtype=np.uint8)
        
        normalized = (raster - raster_min) / (raster_max - raster_min)
        return (normalized * 255).astype(np.uint8)
    
    def preprocess_image(self):
        """Reproject and clip the input raster."""
        print(f"[{self.vector_basename}] Preprocessing image...")
        
        preprocessor = Preprocessing()
        preprocessor.reproject(self.raster_path)
        
        # Clip with buffer for processing
        self.clipped_data, self.clipped_transform = preprocessor.clipraster(
            vector_data=self.vector_gdf,
            buffer_dist=25,
            bbox=False
        )

        display_image, self.display_transform = preprocessor.clipraster(
            vector_data=self.vector_gdf,
            bbox=True
        )
        
        # Save visualization
        self.display_image = self.normalize_raster(display_image)
        self._save_figure(
            display_image,
            self.display_transform,
            "01_clipped_data.png",
            "Input Data with Planned FMR"
        )
        
        return preprocessor
    
    def generate_binary_raster(self):
        """Apply filters and morphological operations to create binary raster."""
        print(f"[{self.vector_basename}] Generating binary raster...")
        
        filter_obj = Filters()
        morph = Morph()
        
        # BSG processing: warmth enhancement + stretch
        warm_raster = filter_obj.enhance_image_warmth(self.clipped_data)
        stretch_raster = filter_obj.enhance_linear_stretch(self.clipped_data)
        
        morph_warm = morph.process(warm_raster, a=4, b=3, ite1=2, ite2=3)
        morph_stretch = morph.process(stretch_raster, a=4, b=3, ite1=2, ite2=3)
        
        binary_raster = np.logical_or(morph_warm, morph_stretch)
        
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
        
        # Save visualization
        self._save_binary_figure(
            self.final_binary_raster,
            "02_binary_raster.png",
            "Binary Road Mask"
        )
        
    def measure_width(self, interval=3, tolerance=0.15, resolution=0.3):
        """Generate transects and measure road width."""
        print(f"[{self.vector_basename}] Measuring road width...")
        
        measure = MeasureWidth(
            self.final_binary_raster,
            self.final_binary_transform,
            self.vector_gdf
        )
        
        self.transects = measure.process(int=interval, tol=tolerance, res=resolution)
        road_mean_width = self.transects['width'].mean()
        
        self.results['mean_width'] = float(road_mean_width)
        self.results['width_std'] = float(self.transects['width'].std())
        self.results['width_min'] = float(self.transects['width'].min())
        self.results['width_max'] = float(self.transects['width'].max())
        
        # Save visualization
        display_image = self.normalize_raster(self.clipped_data)
        self._save_figure(
            display_image,
            self.final_binary_transform,
            "03_transects.png",
            "Width Measurement Transects",
            overlay_gdf=self.transects,
            overlay_color='red'
        )
        
        return road_mean_width
    
    def extract_centerline(self, preprocessor, spacing=3):
        """Extract road centerline from binary raster."""
        print(f"[{self.vector_basename}] Extracting centerline...")
        
        # Clip binary raster tightly for centerline extraction
        binary_clipped, self.binary_transform = preprocessor.clipraster(
            raster_data=self.final_binary_raster.astype(np.uint8),
            vector_data=self.vector_gdf,
            transform=self.clipped_transform,
            buffer_dist=3
        )
        self.binary_clipped = np.squeeze(binary_clipped)
        
        # Generate centerline
        self.centerline = measure_line(
            self.binary_clipped,
            self.binary_transform,
            spacing=spacing,
            crs="EPSG:32651"
        )
        
        # Calculate metrics
        actual_length = self.centerline.length.values[0]
        planned_length = self.vector_gdf.to_crs("EPSG:32651").length.sum()
        
        self.results['planned_length_m'] = float(planned_length)
        self.results['actual_length_m'] = float(actual_length)
        self.results['progress_percent'] = float((actual_length / planned_length) * 100)
        self.results['status'] = self._determine_status(
            self.results['progress_percent']
        )
        
        # Save visualizations
        self._save_figure(
            self.clipped_data,
            self.clipped_transform,
            "04_centerline.png",
            "Extracted Centerline",
            overlay_gdf=self.centerline,
            overlay_color='red'
        )
        
        self._save_comparison_figure()
        
    def generate_polygon(self, road_mean_width):
        """Generate road polygon from centerline and mean width."""
        print(f"[{self.vector_basename}] Generating road polygon...")
        
        # Use the same clipped binary as centerline
        preprocessor = Preprocessing()
        preprocessor.reproject(self.raster_path)
        
        binary_clipped, binary_transform = preprocessor.clipraster(
            raster_data=self.final_binary_raster.astype(np.uint8),
            vector_data=self.vector_gdf,
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
        
        # Calculate area
        self.results['polygon_area_sqm'] = float(self.road_polygon.geometry.area.sum())
        
        # Save visualization
        display_image = self.normalize_raster(self.clipped_data)
        self._save_figure(
            display_image,
            self.clipped_transform,
            "05_road_polygon.png",
            "Road Polygon",
            overlay_gdf=self.road_polygon,
            overlay_color='red',
            overlay_alpha=0.5
        )
    
    def export_results(self, geojson_centerline_path, geojson_polygon_path):
        """Export all results to files."""
        print(f"[{self.vector_basename}] Exporting results...")
        
        # Export raster products
        clipped_path = os.path.join(self.output_folder, f"{self.vector_basename}_clipped.tif")
        export(self.display_image, clipped_path, 'raster', 'EPSG:32651', 
               raster_transform=self.display_transform)
        
        binary_path = os.path.join(self.output_folder, f"{self.vector_basename}_binary.tif")
        export(self.binary_clipped, binary_path, 'raster', 'EPSG:32651',
               raster_transform=self.binary_transform)
        
        # Export vector products as individual shapefiles
        centerline_shp = os.path.join(self.output_folder, f"{self.vector_basename}_centerline.shp")
        export(self.centerline, centerline_shp, 'vector', 'EPSG:32651')
        
        polygon_shp = os.path.join(self.output_folder, f"{self.vector_basename}_polygon.shp")
        export(self.road_polygon, polygon_shp, 'vector', 'EPSG:32651')
        
        transects_shp = os.path.join(self.output_folder, f"{self.vector_basename}_transects.shp")
        export(self.transects, transects_shp, 'vector', 'EPSG:32651')
        
        # Prepare data for GeoJSON output
        self._export_to_geojson(geojson_centerline_path, geojson_polygon_path)
        
        print(f"[{self.vector_basename}] Export complete!")
        
        return self.results
    
    def _export_to_geojson(self, geojson_centerline_path, geojson_polygon_path):
        """Export centerline and polygon to separate GeoJSON files."""
        
        # Prepare centerline feature
        centerline_feature = self.centerline.copy()
        centerline_feature['fmr_id'] = self.vector_basename
        centerline_feature['timestamp'] = str(self.results['timestamp'])
        centerline_feature['length_m'] = self.results['actual_length_m']
        centerline_feature['progress_percent'] = self.results['progress_percent']
        centerline_feature['status'] = self.results['status']
        centerline_feature['mean_width_m'] = self.results['mean_width']
        centerline_feature['image_type'] = self.results['image_type']
        centerline_feature = centerline_feature.to_crs("EPSG:4326")
        
        # Prepare polygon feature
        polygon_feature = self.road_polygon.copy()
        polygon_feature['fmr_id'] = self.vector_basename
        polygon_feature['timestamp'] = str(self.results['timestamp'])
        polygon_feature['area_sqm'] = self.results['polygon_area_sqm']
        polygon_feature['mean_width_m'] = self.results['mean_width']
        polygon_feature['image_type'] = self.results['image_type']
        polygon_feature = polygon_feature.to_crs("EPSG:4326")
        
        # Handle centerlines
        if os.path.exists(geojson_centerline_path):
            existing_centerlines = gpd.read_file(geojson_centerline_path)
            existing_centerlines = existing_centerlines[existing_centerlines['fmr_id'] != self.vector_basename]
            updated_centerlines = gpd.GeoDataFrame(
                pd.concat([existing_centerlines, centerline_feature], ignore_index=True),
                crs="EPSG:4326"
            )
        else:
            updated_centerlines = centerline_feature
        
        # Handle polygons
        if os.path.exists(geojson_polygon_path):
            existing_polygons = gpd.read_file(geojson_polygon_path)
            existing_polygons = existing_polygons[existing_polygons['fmr_id'] != self.vector_basename]
            updated_polygons = gpd.GeoDataFrame(
                pd.concat([existing_polygons, polygon_feature], ignore_index=True),
                crs="EPSG:4326"
            )
        else:
            updated_polygons = polygon_feature
        
        # Convert all timestamps to strings before saving
        if 'timestamp' in updated_centerlines.columns:
            updated_centerlines['timestamp'] = updated_centerlines['timestamp'].astype(str)
        if 'timestamp' in updated_polygons.columns:
            updated_polygons['timestamp'] = updated_polygons['timestamp'].astype(str)
        
        # Save both files
        updated_centerlines.to_file(geojson_centerline_path, driver='GeoJSON')
        updated_polygons.to_file(geojson_polygon_path, driver='GeoJSON')
        
        print(f"[{self.vector_basename}] Updated centerlines: {geojson_centerline_path}")
        print(f"[{self.vector_basename}] Updated polygons: {geojson_polygon_path}")
    
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
    
    def _save_figure(self, raster_data, transform, filename, title,
                     overlay_gdf=None, overlay_color='red', overlay_alpha=1.0):
        """Helper to save visualization figures."""
        fig, ax = plt.subplots(figsize=(10, 10))
        show(raster_data, ax=ax, transform=transform)
        
        if overlay_gdf is not None:
            overlay_gdf.plot(ax=ax, color=overlay_color, 
                           edgecolor=None, linewidth=1, alpha=overlay_alpha)
        
        self.vector_gdf.plot(ax=ax, color='blue', linewidth=1, 
                            label='Planned FMR', alpha=0.7)
        
        ax.set_title(title)
        ax.axis('off')
        ax.legend()
        
        filepath = os.path.join(self.image_output_path, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _save_binary_figure(self, binary_data, filename, title):
        """Helper to save binary raster figures."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(binary_data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
        filepath = os.path.join(self.image_output_path, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
    
    def _save_comparison_figure(self):
        """Save comparison figure showing planned vs detected FMR."""
        fig, ax = plt.subplots(figsize=(10, 10))
        show(self.clipped_data, ax=ax, transform=self.clipped_transform)
        
        self.vector_gdf.plot(ax=ax, color='blue', linewidth=1.5, 
                            label='Planned FMR')
        self.centerline.plot(ax=ax, color='red', linewidth=1.5, 
                            label='Detected FMR')
        
        ax.axis('off')
        ax.legend(loc='upper right')
        
        filepath = os.path.join(self.image_output_path, "06_comparison.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
    
    def run(self, geojson_output_dir):
        """Execute the full pipeline."""
        print(f"\n{'='*60}")
        print(f"Processing: {self.vector_basename}")
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

            centerline_path = os.path.join(geojson_output_dir, "fmr_centerlines.geojson")
            polygon_path = os.path.join(geojson_output_dir, "fmr_polygons.geojson")
    
            results = self.export_results(centerline_path, polygon_path)
        
            print(f"\n Processing complete for {self.vector_basename}")
            print(f"  - Status: {results['status']}")
            print(f"  - Progress: {results['progress_percent']:.1f}%")
            print(f"  - Mean Width: {results['mean_width']:.2f}m")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Error processing {self.vector_basename}: {str(e)}")
            raise


def main():
    """Main execution function."""
    
    # Configuration
    GEOJSON_OUTPUT = r"C:\Users\user-307E4B3400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\.BAFE - FMR (DPWH)\web_output"
    
    # BSG Example
    raster_path = r"C:\Users\user-307E4B3400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\GUI\Raster images\BSG-115-20240601-230820-244317407-Tiff.tif"
    vector_path = r"C:\Users\user-307E4B3400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\.BAFE - FMR (DPWH)\FMR-vectors\FMR-317.geojson"
    output_base = r"C:\Users\user-307E4B3400\OneDrive - Philippine Space Agency\SDMAD_SHARED\PROJECTS\SAKA\FMR\.BAFE - FMR (DPWH)\web_output\outputs"
    
    # Create pipeline and run
    pipeline = RoadDelineationPipeline(
        raster_path=raster_path,
        vector_path=vector_path,
        output_base_dir=output_base,
    )
    
    results = pipeline.run(GEOJSON_OUTPUT)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()