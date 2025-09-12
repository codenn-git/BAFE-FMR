#util v.3.0 focused on revising the measure_line function
## mostly on smoothing the centerline
## then creating a polygon from the calculate mean Width
import rasterio
import rasterio.transform
from rasterio.plot import show
from rasterio.mask import mask
import geopandas as gpd
import shapely
from rasterio.features import shapes
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from scipy.spatial.distance import cdist
import skimage
import skimage.morphology
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import mpl_interactions
import cv2
import shapely
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.measure import label


class Preprocessing:
    '''Class for preprocessing the raster'''
    def __init__(self, pneo=False):
        self.raster_path = None
        self.vector_gdf = None
        self.crs = None
        self.pneo = pneo

        self._rep_data = None
        self._rep_trans = None
        self._rep_crs = None

        self._clipped_data = None
        self._clipped_transform = None

    def reproject(self, raster_path, target_crs="EPSG:32651", resampling=rasterio.enums.Resampling.nearest):
        self.crs = target_crs
        self.raster_path = raster_path

        if self.pneo:
            with rasterio.open(self.raster_path) as src:
                with rasterio.vrt.WarpedVRT(src, crs=target_crs, resampling=resampling) as vrt:
                    self._rep_data = vrt.read([1,2,3,4])
                    self._rep_trans = vrt.transform
                    self._rep_crs = vrt.crs
                    bounds = vrt.bounds
        else:
            with rasterio.open(self.raster_path) as src:
                with rasterio.vrt.WarpedVRT(src, crs=target_crs, resampling=rasterio.enums.Resampling.nearest) as vrt:
                    self._rep_data = vrt.read()
                    self._rep_trans = vrt.transform
                    self._rep_crs = vrt.crs
                    bounds = vrt.bounds

        return self._rep_data, self._rep_trans, self._rep_crs, bounds

    def clipraster(self, raster_data=None, vector_data=None, transform=None, buffer_dist = 15, bbox = False):
        '''Clip the raster data using input vector data. If no vector data is provided, output will be the whole image (reprojected image)
            Args:
                raster_data: Raster data to be clipped
                vector_data: GDF, Vector data to clip the raster
                transform: Transform of the raster data
                buffer_dist: Buffer distance for the vector data
                bbox: If True, use bounding box for clipping'''
        self.vector_gdf = vector_data.to_crs(self.crs)

        if raster_data is None or transform is None:
            if self._rep_data is None or self._rep_trans is None:
                raise ValueError("Reproject has not been performed. Call reproject() first.")
            else:
                raster_data = self._rep_data
                transform = self._rep_trans

        if self.vector_gdf is None: 
            self._clipped_data = self._rep_data
            self._clipped_transform = self._rep_trans

            return self._clipped_data, self._clipped_transform
        
        else:
            self.vector_gdf = vector_data.to_crs(self.crs)
            centerline = self.vector_gdf
        
        buffered_lines = []
        for geometry in centerline.geometry:
            if isinstance(geometry, shapely.geometry.MultiLineString):
                for line in geometry.geoms:
                    buffered_lines.append(line.buffer(buffer_dist))
            elif isinstance(geometry, shapely.geometry.LineString):
                buffered_lines.append(geometry.buffer(buffer_dist))
            else:
                raise ValueError("Centerline not properly formatted") 

        buffered_lines_gdf = gpd.GeoDataFrame({"geometry": buffered_lines}, crs=self.crs)
    
        if bbox:
            bounding_box = buffered_lines_gdf.total_bounds
            bbox_polygon = shapely.geometry.box(*bounding_box)
            clipping_gdf = gpd.GeoDataFrame({"geometry": [bbox_polygon]}, crs=self.crs)
        else:
            clipping_gdf = buffered_lines_gdf

        clipping_geom = [geom.__geo_interface__ for geom in clipping_gdf.geometry]

        # Perform masking
        count = raster_data.shape[0] if len(raster_data.shape) == 3 else 1
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=raster_data.shape[-2],
                width=raster_data.shape[-1],
                count=count,
                dtype=raster_data.dtype,
                transform=transform,
                crs=self.crs,
            ) as dataset:
                if count == 1:
                    dataset.write(raster_data, 1)
                else:
                    for i in range(count):
                        dataset.write(raster_data[i], i + 1)

                # Debug masking
                try:
                    self._clipped_data, self._clipped_transform = mask(dataset, clipping_geom, crop=True)
                except ValueError as e:
                    print("Error during masking:", str(e))
                    raise

        return self._clipped_data, self._clipped_transform
    
    def display(self):
        vector = self.vector_gdf

        fig, ax = plt.subplots(figsize=(10, 10))

        if self._clipped_data is None:
            raise ValueError("Clipped data has not been generated. Call 'clipraster()' first.")
        else:
            data = self._clipped_data
            
        show(data, ax=ax, transform=self._clipped_transform)
        vector.plot(ax=ax, color='red', edgecolor=None, linewidth=1)
        vector.plot(ax=ax, color='red', edgecolor=None, linewidth=1, label='FMR')
        ax.legend(loc='upper right')

        plt.title(f"Check whether the FMR vector is aligned with the image.")
        plt.axis('off')
        plt.show()

class Filters:
    def __init__(self, pneo=False):
        self.pneo = pneo

    # @njit    
    def enhance_image_warmth(self, raster_data):
        '''Enhances the warmth of an image's RGB bands
        args:
            raster_data: numpy array, the input raster image with shape (bands, height, width)
        Returns:
            enhanced_data: numpy array, the enhanced raster image with shape (bands, height, width)
        '''

        if raster_data.dtype == np.uint16:
            raster_data = (raster_data / 65535.0 * 255).astype(np.uint8)
        
        else:
            raster_data = raster_data
            
        raster_data = np.moveaxis(raster_data, 0, -1) 
        enhanced_data = raster_data.copy()

        # Create masks for different conditions
        r = enhanced_data[:, :, 0]
        g = enhanced_data[:, :, 1]
        b = enhanced_data[:, :, 2]

        # Apply transformations for red channel
        r_mask_1 = (r < 30)
        r_mask_2 = (r >= 30) & (r < 100)
        r_mask_3 = (r >= 100) & (r < 175)
        r_mask_4 = (r >= 175) & (r < 255)

        r[r_mask_1] += 35
        r[r_mask_2] += 20
        r[r_mask_3] += 15
        r[r_mask_4] += 7

        # Apply transformations for green channel
        g_mask_1 = (g < 30)
        g_mask_2 = (g >= 30) & (g < 50)
        g_mask_3 = (g >= 50) & (g < 80)
        g_mask_4 = (g >= 80) & (g < 102)

        g[g_mask_1] += 30
        g[g_mask_2] += 20
        g[g_mask_3] += 15
        g[g_mask_4] = 90

        # Apply transformations for blue channel
        b_mask_1 = (b > 150)
        b_mask_2 = (b > 70) & (b <= 150)
        b_mask_3 = (b > 30) & (b <= 70)
        b_mask_4 = (b > 0) & (b <= 30)

        b[b_mask_1] -= 90
        b[b_mask_2] -= 40
        b[b_mask_3] -= 20
        b[b_mask_4] = 15

        # Update the enhanced_data array
        enhanced_data[:, :, 0] = r
        enhanced_data[:, :, 1] = g
        enhanced_data[:, :, 2] = b
        
        enhanced_data = (enhanced_data.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
        enhanced_data = np.moveaxis(enhanced_data, -1, 0)

        return enhanced_data
    
    def enhance_linear_stretch(self, raster_data, lower_percent=98, upper_percent=100):
        """
        Apply linear stretching to enhance the image contrast.

        Parameters:
        - image: numpy array, the input raster image.
        - lower_percent: float, lower percentile to saturate.
        - upper_percent: float, upper percentile to saturate.

        Returns:
        - Stretched image as a numpy array.
        """
        in_min = np.percentile(raster_data, lower_percent)
        in_max = np.percentile(raster_data, upper_percent)
        image = np.clip(raster_data, in_min, in_max)
        out_min, out_max = np.min(raster_data), np.max(raster_data)
        stretched_image = (image - in_min) / ((in_max - in_min) * (out_max - out_min))

        return stretched_image
    
    def cielab(self, raster_data):
        raster_data  = np.moveaxis(raster_data[:3, :, :], 0, -1)
        lab = skimage.color.rgb2lab(raster_data)

        return lab
    
class Morph:
    def __init__(self):
        self.raster_data = None

        self._normalized = None
        self._edges = None
        self._thresholded = None
        self._merged = None
        self._morphed = None
        self._normalized_gray = None

    def normalize_band(self, band):
        return (band - band.min()) / (band.max() - band.min())

    # Normalize all bands
    def normalize_raster(self):
        if self.raster_data.shape[0] > 3:
            self.raster_data = self.raster_data[3:] # Use only the first 3 bands for RGB
        else:
            self.raster_data = self.raster_data

        self._normalized = (self.raster_data - self.raster_data.min()) / (self.raster_data.max() - self.raster_data.min())
        
        if self._normalized.shape[0] == 3:  # For RGB
            self.normalized_grayscale = skimage.color.rgb2gray(np.moveaxis(self._normalized, 0, -1))
            self._normalized_gray = self.normalized_grayscale
            
        else:
            self.normalized_grayscale = self._normalized[0]  #if single-band image
            self._normalized_gray = self.normalized_grayscale
            
    def detect_edge(self):
        self._edges = skimage.feature.canny(self.normalized_grayscale, sigma=1.4, low_threshold = 0.1, high_threshold = 0.5)
            
    def threshold_raster(self):
        otsu = skimage.filters.threshold_otsu(self.normalized_grayscale)
        self._thresholded = self.normalized_grayscale > otsu

    def merge(self):
        if self._edges is not None and self._thresholded is not None:
            self._merged = np.logical_and(self._edges, self._thresholded)
        else:
            raise ValueError("Edge detection or thresholding has not been performed yet.")

    def morphology(self, a=5, b=3, ite1=3, ite2=7):
        thresholded = self._thresholded.astype(np.uint8)

        kernel_a = np.ones((a, a), np.uint8)
        canny_dilated = cv2.dilate(self._edges.astype(np.uint8), kernel_a, iterations=ite1)

        kernel_b = np.ones((b,b), np.uint8)
        canny_eroded = cv2.erode(canny_dilated, kernel_b, iterations=ite2).astype(np.uint8)

        ##Merged_0: Merged (T+(T>D>E))
        self._morphed = np.logical_and(thresholded, canny_eroded) 
        
    def process(self, raster_data, a=5, b=3, ite1=3, ite2=7):
        self.raster_data = raster_data

        self.normalize_raster()
        self.detect_edge()
        self.threshold_raster()
        self.morphology(a=a,b=b,ite1=ite1,ite2=ite2)
        
        return self._morphed
    
    def remove_small_islands(self, raster_data, min_size=100):
        '''
        Remove small islands from the raster data.
        This function labels connected regions in the raster data and removes those smaller than `min_size`.
        
        Args:
            raster_data: numpy array, the input raster image with shape (height, width)
            min_size: int, minimum size of the islands to keep
        Returns:
            cleaned: numpy array, the cleaned raster image with shape (height, width)
        '''

        # Label connected regions
        labels = skimage.measure.label(raster_data, connectivity=2)
        
        # Remove small islands
        cleaned = skimage.morphology.remove_small_objects(labels, min_size=min_size)

        return (cleaned > 0).astype(np.uint8)

    def threshold_cielab(self, raster_data):
        lab = raster_data
        lab_normalized = skimage.exposure.rescale_intensity(lab, in_range=(lab.min(), lab.max()), out_range=(0, 1))

        # Get the Otsu threshold
        otsu_threshold = skimage.filters.threshold_otsu(lab_normalized[..., 0])  # Use the L* channel for thresholding

        # Apply the threshold to create a binary image
        binary_image = lab_normalized[..., 0] > otsu_threshold
        self._thresholded = binary_image

        return binary_image

    def display(self, data_type="morphed"):
        if data_type == "normalized":
            if self._normalized is None:
                raise ValueError("Normalized data has not been generated. Call `process()` first.")
            data = self._normalized
        elif data_type == "normalized_gray":
            if self._normalized_gray is None:
                 raise ValueError("Grayscale has not been performed. Call `process()` first.")
            data = self._normalized_gray
        elif data_type == "edges":
            if self._edges is None:
                raise ValueError("Edge detection has not been performed. Call `process()` first.")
            data = self._edges.astype(np.uint8)  # Convert boolean to uint8
        elif data_type == "thresholded":
            if self._thresholded is None:
                raise ValueError("Thresholding has not been performed. Call `process()` first.")
            data = self._thresholded.astype(np.uint8)  # Convert boolean to uint8
        elif data_type == "merged":
            if self._merged is None:
                raise ValueError("Merged result has not been created. Call `process()` first.")
            data = self._merged.astype(np.uint8)  # Convert boolean to uint8
        elif data_type == "morphed":
            if self._morphed is None:
                raise ValueError("Final result has not been created. Call `process()` first.")
            data = self._morphed.astype(np.uint8)  # Convert boolean to uint8
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Choose from 'normalized', 'edges', 'thresholded', 'merged', 'morphed'.")
        
        data = data
        plt.imshow(data, cmap='gray')
        plt.title(f"{data_type}")
        plt.axis('off')
        plt.show()

    @property
    def normalized(self):
        if self._normalized is None:
            raise ValueError("Normalization has not been performed. Call the `process()` method first.")
        return self._normalized

    @property
    def edges(self):
        if self._edges is None:
            raise ValueError("Normalization has not been performed. Call the `process()` method first.")
        return self._edges

    @property
    def thresholded(self):
        if self._thresholded is None:
            raise ValueError("Normalization has not been performed. Call the `process()` method first.")
        return self._thresholded

    @property
    def merged(self):
        if self._merged is None:
            ValueError("Normalization has not been performed. Call the `process()` method first.")
        return self._merged

    @property  
    def output(self):
        if self._morphed is None:
            ValueError("Normalization has not been performed. Call the `process()` method first.")
        return self._morphed

class MeasureWidth:
    def __init__(self, raster_data, raster_transform, vector_gdf, raster_crs="EPSG:32651"):
        self.raster_data = raster_data
        self.transform = raster_transform
        self.crs = raster_crs
        self.centerline = vector_gdf.to_crs(raster_crs)
        self.vectorized_roads = None
        self._transects = None
        self.clipped_transects = None
        
    def create_transects(self, transect_length=10, interval=3):
        """Create transects perpendicular to a centerline at regular intervals."""
        self.centerline = self.centerline.geometry[0]
        
        transects = []
        for i in range(0, int(self.centerline.length), interval):
            point = self.centerline.interpolate(i)
            next_point = self.centerline.interpolate(i + 1)

            # Convert tuples to numpy arrays and ensure they are 2D (x, y)
            point_coords = np.array(point.coords[0][:2])
            next_point_coords = np.array(next_point.coords[0][:2])

            normal = next_point_coords - point_coords
            normal = np.array([-normal[1], normal[0]])  # Rotate 90 degrees to get normal vector
            normal = normal / np.linalg.norm(normal) # Normalize the vector
            
            transect_start = point_coords + normal * -transect_length
            transect_end = point_coords + normal * transect_length

            transects.append(shapely.geometry.LineString([transect_start, transect_end]))

        self._transects = gpd.GeoDataFrame(geometry=transects, crs=self.crs)
        return self._transects

    def vectorize_roads(self, smooth=True, tolerance=None, resolution=None):
        """Vectorize the road areas (where raster value is 1) and optionally smooth the edges."""
        # Mask out non-road areas
        road_mask = self.raster_data == 1
        if np.sum(road_mask) == 0:
            print("No roads found in the raster data.")
            return None  # No roads to vectorize

        # Vectorize the road mask using rasterio.features.shapes
        shapes_generator = rasterio.features.shapes(self.raster_data.astype(np.uint8), mask=road_mask, transform=self.transform)
    
        # Create GeoDataFrame from vectorized shapes
        if tolerance is None:
            tolerance = 1.0

        geometries = []
        for geom, value in shapes_generator:
            if value == 1:  # Only keep the shapes with value 1 (road areas)
                geom_shape = shapely.geometry.shape(geom)
                # Apply smoothing if enabled
                if smooth:
                    geom_shape = geom_shape.buffer(tolerance, resolution=resolution).buffer(-tolerance, resolution=resolution).simplify(tolerance, preserve_topology=True)
                geometries.append(geom_shape)
        
        if len(geometries) == 0:
            print("No vectorized roads found.")
            return None  # No geometries were created
        
        # Create a GeoDataFrame to store the vectorized roads
        self.vectorized_roads = gpd.GeoDataFrame(geometry=geometries, crs=self.crs)
        return self.vectorized_roads

    def clip_transects(self):
        """Clip transects using the vectorized roads and save the result."""
        if self.vectorized_roads is None:
            raise ValueError("Roads not vectorized. Run vectorize_roads() first.")

        # Clip transects with the vectorized roads
        clipped = gpd.overlay(self._transects, self.vectorized_roads, how='intersection')
        clipped = clipped.explode(index_parts=True)
        clipped = clipped.explode(index_parts=True)

        self.clipped_transects = clipped
        self.clipped_transects['width'] = clipped.geometry.length

        return self.clipped_transects[['geometry', 'width']]
    
    def filter_transects(self):
        if self.clipped_transects is None:
            raise ValueError("Transects not measured. Run clip_transects() first.")

        self.clipped_transects = self.clipped_transects[(self.clipped_transects["width"] >= 3.5) & (self.clipped_transects["width"] <= 8)]

        return self.clipped_transects[['geometry', 'width']]

        ##mean based filter
        # if self.clipped_transects["width"].mean() < 6 and self.clipped_transects["width"].mean() > 3.5:
        #     self.clipped_transects = self.clipped_transects[(self.clipped_transects["width"] >= 3.7) & (self.clipped_transects["width"] <= 6)]
        # else:                                                    
        #     self.clipped_transects = self.clipped_transects[(self.clipped_transects["width"] >= 4) & (self.clipped_transects["width"] <= 8)]

    def process(self, int, tol, res):
        self.create_transects(interval=int)
        self.vectorize_roads(tolerance=tol, resolution=res)
        self.clip_transects()
        self.filter_transects()
        return self.clipped_transects[['geometry', 'width']]


    def export(self, output_path, gdf=None):
         # Ensure the output folder exists
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if self.clipped_transects is None:
            raise ValueError("Transects not measured. Run process() first.")
        
        if gdf is None:
            gdf = self.clipped_transects

        gdf.to_file(output_path, crs=self.crs)

        return print(f"Road width exported successfully to {output_path}")
    
    @property
    def vectorized(self):
        if self.vectorized_roads is None:
            raise ValueError("Road not vectorized. Run process() first.")
        return self.vectorized_roads
    
    @property
    def transects(self):
        if self.clipped_transects is None:
            raise ValueError("Transects not measured. Run process() first.")
        
        return self.clipped_transects
    
    def display(self, vector_data, raster, ax):
        show(raster, ax=ax, transform=self.transform)
        vector_data.plot(ax=ax, color='red', edgecolor=None, linewidth=1)
        ax.set_title("Generated Transects")
        ax.axis("off")
        
        return plt.show()

class CenterlineExtractor:
    """
    Extract and process centerlines from skeleton rasters.
    """

    def __init__(self, raster_array=None, transform=None, crs=None):
        raster_array = skeletonize(raster_array)
        self.raster = raster_array.astype(np.uint8)
        self.transform = transform
        self.crs = crs

   
    # PREPROCESSING #===========================================================================

    def preprocess(self, ensure_binary=True, apply_thinning=True):
        if ensure_binary:
            self.raster = (self.raster > 0).astype(np.uint8)
        if apply_thinning:
            self.raster = thin(self.raster).astype(np.uint8)
        return self.raster

    # =============================================================================
    # CENTERLINE EXTRACTION
    # =============================================================================

    def extract_centerlines(self, method='pixel_tracing'):
        if method == 'pixel_tracing':
            return self._extract_by_pixel_tracing()
        elif method == 'vectorization':
            return self._extract_by_vectorization()
        elif method == 'endpoint_detection':
            return self._extract_by_endpoints()
        else:
            raise ValueError("Method must be 'pixel_tracing', 'vectorization', or 'endpoint_detection'")

    def _extract_by_pixel_tracing(self):
        skeleton_pixels = np.where(self.raster > 0)
        if len(skeleton_pixels[0]) == 0:
            return gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)

        labeled = label(self.raster > 0, connectivity=2)
        lines = []
        for component_id in range(1, labeled.max() + 1):
            component_mask = labeled == component_id
            component_coords = np.where(component_mask)
            if len(component_coords[0]) < 2:
                continue
            real_coords = []
            for i in range(len(component_coords[0])):
                row, col = component_coords[0][i], component_coords[1][i]
                x, y = rasterio.transform.xy(self.transform, row, col)
                real_coords.append((x, y))
            if len(real_coords) >= 2:
                sorted_coords = self._sort_coordinates(real_coords)
                if len(sorted_coords) >= 2:
                    lines.append(LineString(sorted_coords))
        return gpd.GeoDataFrame(geometry=lines, crs=self.crs) if lines else gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)

    def _extract_by_vectorization(self):
        mask = self.raster > 0
        shapes_gen = shapes(self.raster.astype(np.int32), mask=mask, transform=self.transform)
        geometries = [geom for geom, value in shapes_gen if value > 0]
        return gpd.GeoDataFrame(geometry=geometries, crs=self.crs) if geometries else gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)

    def _extract_by_endpoints(self):
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        skeleton_conv = ndimage.convolve(self.raster.astype(float), kernel, mode='constant')
        labeled = label(self.raster > 0, connectivity=2)
        lines = []
        for component_id in range(1, labeled.max() + 1):
            component_mask = labeled == component_id
            component_coords = np.where(component_mask)
            if len(component_coords[0]) >= 2:
                real_coords = []
                for i in range(len(component_coords[0])):
                    row, col = component_coords[0][i], component_coords[1][i]
                    x, y = rasterio.transform.xy(self.transform, row, col)
                    real_coords.append((x, y))
                if len(real_coords) >= 2:
                    sorted_coords = self._sort_coordinates(real_coords)
                    lines.append(LineString(sorted_coords))
        return gpd.GeoDataFrame(geometry=lines, crs=self.crs) if lines else gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)

    def _sort_coordinates(self, coords):
        if len(coords) <= 2:
            return coords
        points = [Point(x, y) for x, y in coords]
        max_dist = 0
        start_idx = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = points[i].distance(points[j])
                if dist > max_dist:
                    max_dist = dist
                    start_idx = i
        start_point = points[start_idx]
        sorted_indices = sorted(range(len(points)), key=lambda i: start_point.distance(points[i]))
        return [coords[i] for i in sorted_indices]

    # =============================================================================
    # LINE CONNECTION
    # =============================================================================

    def connect_lines(self, gdf, tolerance=10.0, method='nearest'):
        if len(gdf) <= 1:
            return gdf
        lines = list(gdf.geometry)
        if method == 'nearest':
            connected = self._connect_nearest(lines, tolerance)
        elif method == 'sequential':
            connected = self._connect_sequential(lines)
        elif method == 'mst':
            connected = self._connect_mst(lines, tolerance)
        else:
            raise ValueError("Method must be 'nearest', 'sequential', or 'mst'")
        if connected:
            return gpd.GeoDataFrame(geometry=[connected], crs=gdf.crs)
        else:
            return gdf

    def _connect_nearest(self, lines, tolerance):
        if not lines:
            return None
        connected_coords = list(lines[0].coords)
        remaining_lines = lines[1:]
        while remaining_lines:
            current_end = Point(connected_coords[-1])
            best_line = None
            best_distance = float('inf')
            best_idx = -1
            best_reverse = False
            for i, line in enumerate(remaining_lines):
                line_start = Point(line.coords[0])
                line_end = Point(line.coords[-1])
                dist_to_start = current_end.distance(line_start)
                dist_to_end = current_end.distance(line_end)
                if dist_to_start < best_distance:
                    best_distance = dist_to_start
                    best_line = line
                    best_idx = i
                    best_reverse = False
                if dist_to_end < best_distance:
                    best_distance = dist_to_end
                    best_line = line
                    best_idx = i
                    best_reverse = True
            if best_line and (best_distance <= tolerance or len(remaining_lines) == len(lines) - 1):
                line_coords = list(best_line.coords)
                if best_reverse:
                    line_coords = line_coords[::-1]
                if best_distance > 0:
                    connected_coords.append(line_coords[0])
                connected_coords.extend(line_coords)
                remaining_lines.pop(best_idx)
            else:
                break
        return LineString(connected_coords) if len(connected_coords) >= 2 else None

    def _connect_sequential(self, lines):
        if not lines:
            return None
        connected_coords = list(lines[0].coords)
        for line in lines[1:]:
            line_coords = list(line.coords)
            current_end = Point(connected_coords[-1])
            if current_end.distance(Point(line_coords[-1])) < current_end.distance(Point(line_coords[0])):
                line_coords = line_coords[::-1]
            connected_coords.extend(line_coords)
        return LineString(connected_coords)

    def _connect_mst(self, lines, tolerance):
        """Connect lines using a true Minimum Spanning Tree approach"""
        if not lines:
            return None

        # Build list of endpoints
        endpoints = []
        for idx, line in enumerate(lines):
            endpoints.append((idx, 0, Point(line.coords[0])))  # start point
            endpoints.append((idx, 1, Point(line.coords[-1]))) # end point

        # Build full distance matrix between endpoints
        n = len(endpoints)
        dist_matrix = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                if endpoints[i][0] != endpoints[j][0]:  # don't connect endpoints of same line
                    d = endpoints[i][2].distance(endpoints[j][2])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

        try:
            from scipy.sparse.csgraph import minimum_spanning_tree
            mst = minimum_spanning_tree(dist_matrix)
            mst = mst.toarray()
        except ImportError:
            return self._connect_nearest(lines, tolerance)

        # Find pairs to connect based on MST edges below tolerance
        connections = []
        for i in range(n):
            for j in range(n):
                if mst[i, j] != 0 and mst[i, j] <= tolerance:
                    connections.append((i, j))

        # Build connected coordinates following MST edges
        used = [False] * len(lines)
        connected_coords = []

        for idx, line in enumerate(lines):
            if not used[idx]:
                coords = list(line.coords)
                used[idx] = True
                # Find edges connected to this line and append coordinates
                for (i, j) in connections:
                    li, si, pi = endpoints[i]
                    lj, sj, pj = endpoints[j]
                    if li == idx or lj == idx:
                        other_idx = lj if li == idx else li
                        if not used[other_idx]:
                            other_line = lines[other_idx]
                            other_coords = list(other_line.coords)
                            if sj == 1:
                                other_coords = other_coords[::-1]  # flip if needed
                            coords.extend(other_coords)
                            used[other_idx] = True
                connected_coords.extend(coords)

        return LineString(connected_coords) if len(connected_coords) >= 2 else None

    # =============================================================================
    # SMOOTHING
    # =============================================================================

    def smooth_lines(self, gdf, method='spline', **kwargs):
        if len(gdf) == 0:
            return gdf
        smoothed_geometries = []
        for geom in gdf.geometry:
            if method == 'spline':
                smoothed = self._smooth_spline(geom, **kwargs)
            elif method == 'douglas_peucker':
                smoothed = self._smooth_douglas_peucker(geom, **kwargs)
            elif method == 'moving_average':
                smoothed = self._smooth_moving_average(geom, **kwargs)
            elif method == 'gaussian':
                smoothed = self._smooth_gaussian(geom, **kwargs)
            else:
                raise ValueError("Unsupported smoothing method")
            if smoothed:
                smoothed_geometries.append(smoothed)
        return gpd.GeoDataFrame(geometry=smoothed_geometries, crs=gdf.crs) if smoothed_geometries else gdf

    def _smooth_spline(self, line, smoothing_factor=0.1, num_points=None):
        try:
            from scipy.interpolate import splprep, splev
        except ImportError:
            return self._smooth_douglas_peucker(line, tolerance=1.0)
        coords = np.array(line.coords)
        if len(coords) < 4:
            return line
        unique_coords = []
        for coord in coords:
            if not unique_coords or not np.allclose(coord, unique_coords[-1], atol=1e-10):
                unique_coords.append(coord)
        if len(unique_coords) < 4:
            return line
        unique_coords = np.array(unique_coords)
        x, y = unique_coords[:, 0], unique_coords[:, 1]
        try:
            tck, u = splprep([x, y], s=smoothing_factor * len(x), k=min(3, len(x) - 1))
            if num_points is None:
                num_points = max(len(coords), 100)
            u_new = np.linspace(0, 1, num_points)
            smooth_coords = splev(u_new, tck)
            return LineString(list(zip(smooth_coords[0], smooth_coords[1])))
        except:
            return line

    def _smooth_douglas_peucker(self, line, tolerance=1.0):
        return line.simplify(tolerance, preserve_topology=True)

    def _smooth_moving_average(self, line, window_size=5):
        coords = np.array(line.coords)
        if len(coords) <= window_size:
            return line
        smoothed_coords = []
        for i in range(len(coords)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(coords), i + window_size // 2 + 1)
            window_coords = coords[start_idx:end_idx]
            avg_coord = np.mean(window_coords, axis=0)
            smoothed_coords.append(avg_coord)
        return LineString(smoothed_coords)
        coords = list(line.coords)
        for _ in range(iterations):
            if len(coords) < 3:
                break
            new_coords = [coords[0]]
            for i in range(len(coords) - 1):
                p1, p2 = np.array(coords[i]), np.array(coords[i + 1])
                q1 = p1 + 0.25 * (p2 - p1)
                q2 = p1 + 0.75 * (p2 - p1)
                new_coords.extend([q1, q2])
            new_coords.append(coords[-1])
            coords = new_coords
        return LineString(coords)

    def _smooth_gaussian(self, line, sigma=1.0):
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            return self._smooth_moving_average(line, window_size=5)
        coords = np.array(line.coords)
        if len(coords) < 3:
            return line
        x_smooth = gaussian_filter1d(coords[:, 0], sigma=sigma, mode='nearest')
        y_smooth = gaussian_filter1d(coords[:, 1], sigma=sigma, mode='nearest')
        return LineString(list(zip(x_smooth, y_smooth)))

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def clean_lines(self, gdf):
        if len(gdf) == 0:
            return gdf
        merged_lines = linemerge(list(gdf.geometry))
        if hasattr(merged_lines, 'geoms'):
            geometries = list(merged_lines.geoms)
        else:
            geometries = [merged_lines]
        return gpd.GeoDataFrame(geometry=geometries, crs=gdf.crs)

    def visualize(self, *gdfs, titles=None, figsize=(15, 5)):
        n_plots = len(gdfs) + 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        axes[0].imshow(self.raster, cmap='gray')
        axes[0].set_title('Original Skeleton')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, gdf in enumerate(gdfs):
            ax_idx = i + 1
            if len(gdf) > 0:
                gdf.plot(ax=axes[ax_idx], color=colors[i % len(colors)], linewidth=2)
            title = titles[i] if titles and i < len(titles) else f'Result {i+1}'
            axes[ax_idx].set_title(title)
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].set_xlabel('X Coordinate')
            axes[ax_idx].set_ylabel('Y Coordinate')
        plt.tight_layout()
        plt.show()

    def process_skeleton(self, extract_method='pixel_tracing', connect_lines=True, connection_tolerance=10.0, smooth_method=None, smooth_params=None):
        
        self.preprocess()
        
        centerlines = self.extract_centerlines(method=extract_method)
        centerlines = self.clean_lines(centerlines)
        
        if connect_lines and len(centerlines) > 1:
            connected = self.connect_lines(centerlines, tolerance=connection_tolerance)
        else:
            connected = centerlines
        
        result = connected
        if smooth_method:
            smooth_params = smooth_params or {}
            result = self.smooth_lines(result, method=smooth_method, **smooth_params)

def create_polygon(line_gdf, road_width):
    """
    Buffer connected lines to create road polygons.
    
    Parameters:
    - line_gdf: GeoDataFrame containing connected LineStrings
    - road_width: width of the road in map units (e.g. meters)
    
    Returns:
    - GeoDataFrame containing buffered polygons
    """
    buffered_polygons = line_gdf.geometry.buffer(road_width / 2, cap_style=1, join_style=1)
    return gpd.GeoDataFrame(geometry=buffered_polygons, crs=line_gdf.crs)

def stretch_band(band, lower_percent=2, upper_percent=98):
    ''' 
    For visualization purposes, stretch the band using percentiles.
        Args:
            band: numpy array, the input band to be stretched
            lower_percent: float, lower percentile to use for stretching
            upper_percent: float, upper percentile to use for stretching
        Returns:
            stretched: numpy array, the stretched band
    '''
    lower = np.percentile(band, lower_percent)
    upper = np.percentile(band, upper_percent)
    stretched = np.clip((band - lower) / (upper - lower), 0, 1)
    return stretched

def export(obj, output_path, obj_type, crs, raster_transform=None):
    '''
    Export the GeoDataFrame to a shapefile.
        Args:
            obj: raster or vector (GeoDataFrame) object
            output_path: str, path to save the object
            obj_type: str, type of the object ('raster' or 'vector')
            crs: str, crs
            raster_transform: transform if obj_type is 'raster'
    '''
    if obj_type not in ['raster', 'vector']:
        raise ValueError("obj_type must be either 'raster' or 'vector'.")
            
    # Ensure the output folder exists
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if obj_type == 'raster':
        if raster_transform is None:
            raise ValueError("Exporting rasters need an input raster transform.")

        # Handle both 2D and 3D arrays
        if obj.ndim == 2:
            # 2D array - single band
            height, width = obj.shape
            count = 1
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width,
                               count=count, dtype=obj.dtype, crs=crs, transform=raster_transform) as dst:
                dst.write(obj, 1)
        else:
            # 3D array - multiple bands
            height, width = obj.shape[1], obj.shape[2]
            count = obj.shape[0]
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width,
                               count=count, dtype=obj.dtype, crs=crs, transform=raster_transform) as dst:
                for i in range(count):
                    dst.write(obj[i], i + 1)

        print(f"Raster exported successfully to {output_path}")
        return

    if obj_type == 'vector':
        # For vector objects
        if isinstance(obj, gpd.GeoDataFrame):
            obj.to_file(output_path, driver='GeoJSON')
        else:
            raise ValueError("obj must be a GeoDataFrame for vector export.")

        print(f"Vector exported successfully to {output_path}")
        return