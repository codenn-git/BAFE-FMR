import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import geopandas as gpd
import shapely
import skimage
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import mpl_interactions
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.figure import Figure

class Preprocessing:
    '''Class for preprocessing the raster'''
    def __init__(self, raster_path, vector_gdf, crs="EPSG:32651", pneo=False):
        self.raster_path = raster_path
        self.vector_gdf = vector_gdf.to_crs(crs)
        self.crs = crs
        self.pneo = pneo

        self._rep_data = None
        self._rep_trans = None
        self._rep_crs = None

        self._clipped_data = None
        self._clipped_transform = None

    def reproject(self, target_crs="EPSG:32651", resampling=rasterio.enums.Resampling.nearest):
        if self.pneo:
            with rasterio.open(self.raster_path) as src:
                with rasterio.vrt.WarpedVRT(src, crs=target_crs, resampling=resampling) as vrt:
                    self._rep_data = vrt.read([1,2,3,4])
                    self._rep_trans = vrt.transform
                    self._rep_crs = vrt.crs
        else:
            with rasterio.open(self.raster_path) as src:
                with rasterio.vrt.WarpedVRT(src, crs=target_crs, resampling=rasterio.enums.Resampling.nearest) as vrt:
                    self._rep_data = vrt.read()
                    self._rep_trans = vrt.transform
                    self._rep_crs = vrt.crs

        return self._rep_data, self._rep_trans, self._rep_crs

    def clipraster(self, raster_data=None, vector_data=None, transform=None, buffer_dist = 15, bbox = False):
        '''Clip the raster data using input vector data. If no vector data is provided, output will be the whole image (reprojected image)
            Args:
                raster_data: Raster data to be clipped
                vector_data: GDF, Vector data to clip the raster
                transform: Transform of the raster data
                buffer_dist: Buffer distance for the vector data
                bbox: If True, use bounding box for clipping'''
        
        if raster_data is None or transform is None:
            if self._rep_data is None or self._rep_trans is None:
                raise ValueError("Reproject has not been performed. Call reproject() first.")
            else:
                raster_data = self._rep_data
                transform = self._rep_trans

        if vector_data is None:
            if self.vector_gdf is None: #if vector data is not provided, not cropped image will be returned
                self._clipped_data = self._rep_data
                self._clipped_transform = self._rep_trans

                return self._clipped_data, self._clipped_transform
            else:
                centerline = self.vector_gdf.to_crs(self.crs)
                self.vector_gdf = centerline
        else:
            centerline = vector_data.to_crs(self.crs)

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
        raster_data = np.moveaxis(self.raster_data[:3, :, :], 0, -1)  # Convert to HWC format
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
        
    def create_transects(self, transect_length=5, interval=3):
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

        if self.clipped_transects["width"].mean() < 5 and self.clipped_transects["width"].mean() > 3.8:
            self.clipped_transects = self.clipped_transects[(self.clipped_transects["width"] >= 3.7) & (self.clipped_transects["width"] <= 5.3)]
        else:                                                    
            self.clipped_transects = self.clipped_transects[(self.clipped_transects["width"] >= 4) & (self.clipped_transects["width"] <= 8)]

    def process(self, int, tol, res):
        self.create_transects(interval=int)
        self.vectorize_roads(tolerance=tol, resolution=res)
        self.clip_transects()
        self.filter_transects()
        return self.clipped_transects[['geometry', 'width']]

    def generate_polygon(self):
        transects = self.clipped_transects['geometry']
        # Extract the endpoints of the transects
        left_points = [transect.coords[0] for transect in transects]  # Start points
        right_points = [transect.coords[1] for transect in transects]  # End points

        # Create LineStrings by connecting the endpoints
        left_line = shapely.geometry.LineString(left_points)  # Line along the left side
        right_line = shapely.geometry.LineString(right_points)  # Line along the right side

        # Create a Polygon by combining the left and right lines
        polygon = shapely.geometry.Polygon(left_points + right_points[::-1])  # Reverse right points to close the polygon

        gdf_polygon = gpd.GeoDataFrame({
            'id': [1],  # Unique ID for the polygon
            'geometry': [polygon],
            'description': ['Polygon']  # Optional: Add description
        })

        return gdf_polygon

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
    
    def display(self, raster, ax):
        show(raster, ax=ax, transform=self.transform)
        self.clipped_transects.plot(ax=ax, color='red', edgecolor=None, linewidth=1)
        ax.set_title("Generated Transects")
        ax.axis("off")
        
        return plt.show()
    
def measure_line(raster_data, transform, output_shapefile=None, spacing=3, return_points=False, crs="EPSG:32651"):
    """
    Create a continuous road centerline from left to right from binary raster.
    
    Parameters:
    - raster_data: 2D numpy array (1=road, 0=non-road)
    - transform: affine transform from original raster
    - output_shapefile: optional path to save shapefile (.shp)
    - spacing: sample every N pixels (default=1, all points)
    - return_points: if True, returns both points and line
    
    Returns:
    - GeoDataFrame with centerline (and optionally points)
    """
    # Step 1: Create center points
    skeleton = skimage.morphology.skeletonize(raster_data)
    y_coords, x_coords = np.where(skeleton)
    x_coords = x_coords[::spacing]
    y_coords = y_coords[::spacing]
    
    # Convert to real-world coordinates
    xx, yy = rasterio.transform.xy(transform, y_coords, x_coords)
    points = [shapely.geometry.Point(x, y) for x, y in zip(xx, yy)]
    points_gdf = gpd.GeoDataFrame(
        geometry=points,
        data={'pixel_x': x_coords, 'pixel_y': y_coords},
        crs=crs  # Adjust CRS as needed
    )
    
    # Step 2: Connect points left-to-right
    coords = np.array([[p.x, p.y] for p in points_gdf.geometry])
    sorted_idx = np.argsort(coords[:, 0])
    sorted_coords = coords[sorted_idx]
    
    # Build path using nearest neighbor
    path = []
    remaining_points = sorted_coords.copy()
    current_point = remaining_points[0]
    path.append(current_point)
    remaining_points = np.delete(remaining_points, 0, axis=0)
    
    while len(remaining_points) > 0:
        nbrs = NearestNeighbors(n_neighbors=1).fit(remaining_points)
        distances, indices = nbrs.kneighbors([current_point])
        next_point = remaining_points[indices[0][0]]
        path.append(next_point)
        current_point = next_point
        remaining_points = np.delete(remaining_points, indices[0][0], axis=0)
    
    line = shapely.geometry.LineString(path)
    line_gdf = gpd.GeoDataFrame(geometry=[line], crs=points_gdf.crs)
    
    # Save output if requested
    if output_shapefile:
        line_gdf.to_file(output_shapefile)
        if return_points:
            points_gdf.to_file(output_shapefile.replace('.shp', '_points.shp'))
    
    return (line_gdf, points_gdf) if return_points else line_gdf

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

def export(obj, output_path, obj_type):
    '''
    Export the GeoDataFrame to a shapefile.
        Args:
            obj: raster or vector (GeoDataFrame) object
            output_path: str, path to save the object
            obj_type: str, type of the object ('raster' or 'vector')
    '''
    if obj_type not in ['raster', 'vector']:
        raise ValueError("obj_type must be either 'raster' or 'vector'.")
    
    # Ensure the output folder exists
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if obj_type == 'raster':
        # For raster objects
        with rasterio.open(output_path, 'w', driver='GTiff', height=obj.shape[1], width=obj.shape[2],
                           count=obj.shape[0], dtype=obj.dtype, crs=obj.crs, transform=obj.transform) as dst:
            for i in range(obj.shape[0]):
                dst.write(obj[i], i + 1)

        return print(f"Raster exported successfully to {output_path}")

    if obj_type == 'vector':
        # For vector objects
        if isinstance(obj, gpd.GeoDataFrame):
            obj.to_file(output_path, driver='ESRI Shapefile')
        else:
            raise ValueError("obj must be a GeoDataFrame for vector export.")

        return print(f"Vector exported successfully to {output_path}")