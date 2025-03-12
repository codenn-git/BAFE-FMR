##IMPORTS
# import rasterio
from rasterio.mask import mask
import geopandas as gpd
# import shapely
# import skimage
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from util import Preprocessing, Filters, Morph, MeasureWidth, export
import time

def main():
    start_time = time.time()

    image_type = input("Enter the type of image (BSG or PNEO): ").strip().upper()
    if not image_type:
        print("No image type provided. Exiting.")
        return

    #preprocessing
    raster_path = input("Enter the path to the raster: ").strip()
    vector_path = input("Enter the path to the vector (or press Enter to skip): ").strip()
    
    if not vector_path:
        vector_path = None

    dirname = os.path.dirname(raster_path)
    basename = os.path.basename(dirname)

    if image_type == "BSG":
        preprocessor = Preprocessing(raster_path, vector_path)

    elif image_type == "PNEO":
        preprocessor = Preprocessing(raster_path, vector_path, pneo=True)
        
    preprocessor.reproject()
    clipped_data, clipped_transform = preprocessor.clipraster() #bbox=False

    if image_type == "BSG":
        #Filters: warmth and linear stretch
        filter = Filters(clipped_data)
        warm_raster = filter.enhance_image_warmth()
        stretch_raster = filter.enhance_linear_stretch()

        #Apply erode
        morph_warm = Morph(warm_raster)
        morph_warm.process()

        morph_stretch = Morph(stretch_raster)
        morph_stretch.process()

        #merges the applied morphed warmth and stretch function
        merged_or = np.logical_or(morph_warm.output, morph_stretch.output)
        final_raster = merged_or

        # plt.imshow(final_raster, cmap="gray")
        # plt.axis("off")
        # plt.title("Merged")
        # plt.show()

    elif image_type == "PNEO": #PNEO only needs linear stretch and thresholding
        filter = Filters(clipped_data)
        stretch_raster = filter.enhance_linear_stretch()
        
        morph = Morph(stretch_raster)
        morph.normalize_raster() 
        morph.threshold_raster()
        final_raster = morph.thresholded

    else:
        print("Invalid image type. Please enter either BSG or PNEO.")
        return

    final_clipped_data, final_clipped_transform = preprocessor.clipraster(
                            raster_data = final_raster.astype(np.uint8),
                            transform = clipped_transform,
                            buffer_dist = 3.0)


    final_clipped_data = np.squeeze(final_clipped_data)
    final_clipped_data = cv2.morphologyEx(final_clipped_data.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=4)
    
    plt.imshow(final_clipped_data, cmap="gray")
    plt.axis("off")
    plt.title("Clipped Merged")
    plt.show()

    measure = MeasureWidth(final_clipped_data, final_clipped_transform, vector_path)

    if image_type == "PNEO":
        tol = 0.10 ##considering changing this.
    elif image_type == "BSG":
        tol = 0.4
    else:
        print("Invalid image type. Please enter either BSG or PNEO.")
        return

    measure.process(tolerance=tol)

    ##EXPORTING OUTPUTS
    output_folder = input("Enter path to output folder: ").strip()
    
    if not output_folder:
        print("No output folder provided. Finishing.")
        return

    outpath = os.path.join(output_folder, basename, basename + "-road_final.tif")
    
    export(final_clipped_data,final_clipped_transform, output_path=outpath)

    measure.export(os.path.join(output_folder, basename, basename + "-widths.geojson"))
    print("Width extraction completed.")
    print(measure.transects)

    measure.export(os.path.join(output_folder, basename, basename + "-vectorized_road.geojson"), gdf = measure.vectorized_roads)
    print("Exporting Vectorized road completed.")
    
    measure.vectorized_roads.plot()

    end_time = time.time()
    total_run_time = end_time - start_time
    print(f"Total Run Time: {total_run_time:.2f} seconds")

# Entry point of the script
if __name__ == "__main__":
    main()