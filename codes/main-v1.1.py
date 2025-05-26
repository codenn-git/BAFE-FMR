##IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import os
from util import *
import time
import cv2

def main():
    start_time = time.time()

    # Stretch the bands of the image NOTE: I can use the linear_stretch function from the preprocessor module
    def stretch_band(band, lower_percent=2, upper_percent=98):
        lower = np.percentile(band, lower_percent)
        upper = np.percentile(band, upper_percent)
        stretched = np.clip((band - lower) / (upper - lower), 0, 1)
        return stretched
    
    trk_ext = input("Do you want to track FMR progress or Extract information? (T/E): ").strip().upper()
    
    if trk_ext == "T":
        print("Tracking FMR progress only takes BlackSky images.")

        preprocessor = Preprocessing(raster_path, vector_path)
        preprocessor.reproject()
        clipped_data, clipped_transform = preprocessor.clipraster(buffer_dist = 25, bbox=True)
        preprocessor.display()
        
        interaction = Interaction(clipped_data, vector_path)
        interaction.show()

        output_path = input("Enter path to output folder (enter to skip): ").strip()
        
        if not output_path:
            print("No output folder provided. Exiting.")
            return
        
        interaction.export_line(output_path)

    elif trk_ext == "E":

        image_type = input("Enter the type of image (BSG or PNEO): ").strip().upper()

        #INPUTS
        raster_path = input("Enter the path to the raster: ").strip()
        vector_path = input("Enter the path to the vector (or press Enter to skip): ").strip()

        preprocessor = Preprocessing(raster_path, vector_path)
        preprocessor.reproject()

        if not image_type:
            print("No image type provided. Exiting.")
            return
        elif image_type == "PNEO":
            int, tol, res = 3, 0.15, 0.3 
            clipped_data, clipped_transform = preprocessor.clipraster(bbox=True)

            orig_img = input("Do you want to view the image? (Y/N): ").strip().upper()
            
            if orig_img == "Y":
                img = clipped_data
                img = np.moveaxis(img, 0, -1)
                img = img[:, :, :3]

                # Apply stretching to each band
                img = np.stack([stretch_band(img[:, :, i]) for i in range(3)], axis=-1)

                # Plot the vector over the clipped data
                plt.imshow(img)
                plt.axis("off")
                plt.title("Clipped Image")
                plt.show()

                cont_step = input("Do you want to continue to the next step? (y/n): ").strip().lower()

            elif orig_img == "N":
                print("Moving on to the next step.")
                cont_step = "y"

            else:
                print("Invalid input. Please enter Y or N.")
                return
            
            if cont_step == "n":
                print("Exiting the program.")
                return

            elif cont_step == "y":            
                #apply filter; for PNEO only conversion to CIE-LAB is needed
                filter = Filters(clipped_data)
                cielab = filter.cielab()
                
                morph = Morph(cielab)
                morph.threshold_cielab()
                initial_binary_raster = morph.thresholded

            #if user enters wrong letter, go back to input for cont_step
            else:
                print("Invalid input. Please enter Y or N.")
                return
            
        elif image_type == "BSG":
            int, tol, res = 3, 0.4, 0.3
            clipped_data, clipped_transform = preprocessor.clipraster(buffer_dist=25)

            orig_img = input("Do you want to view the image? (Y/N): ").strip().upper()
            
            if orig_img == "Y":
                img = clipped_data
                img = np.moveaxis(img, 0, -1)
                img = img[:, :, :3]

                # Apply stretching to each band
                img = np.stack([stretch_band(img[:, :, i]) for i in range(3)], axis=-1)

                # Plot the vector over the clipped data
                plt.imshow(img)
                plt.axis("off")
                plt.title("Clipped Image")
                plt.show()

                cont_step = input("Do you want to continue to the next step? (y/n): ").strip().lower()

            elif orig_img == "N":
                cont_step = "y"

            else:
                print("Invalid input. Please enter Y or N.")
                return
            
            if cont_step == "n":
                print("Exiting the program.")
                return
            
            elif cont_step == "y":
                #Filters: warmth and linear stretch
                filter = Filters(clipped_data)
                warm_raster = filter.enhance_image_warmth()
                stretch_raster = filter.enhance_linear_stretch()

                #Apply Morphological Operations
                morph = Morph()
                morph_warm = morph.process(warm_raster)
                morph_stretch = morph.process(stretch_raster)

                #merges the applied morphed warmth and stretch function
                merged_or = np.logical_or(morph_warm, morph_stretch)
                initial_binary_raster = merged_or

            else:
                print("Invalid input. Please enter Y or N.")
                return

        else:
            print("Invalid image type. Please enter either BSG or PNEO.")
            return

        #WIDTH MEASUREMENT
        # final_clipped_data, final_clipped_transform = preprocessor.clipraster(
        #                         raster_data = final_raster.astype(np.uint8),
        #                         transform = clipped_transform,
        #                         bbox=True)
        
        #add small island removal here
        final_binary_transform = clipped_transform
 
        # plt.imshow(final_clipped_data, cmap="gray")
        final_binary_raster = morph.remove_small_islands(initial_binary_raster, min_size=1000)
        final_binary_raster = cv2.morphologyEx(final_binary_raster.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)

        # Plot two rasters side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original clipped data
        axes[0].imshow(initial_binary_raster, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Initial Binary Raster")

        # Final processed raster
        axes[1].imshow(final_binary_raster, cmap="gray")
        axes[1].axis("off")
        axes[1].set_title("Final Binary Raster")

        plt.tight_layout()
        plt.show()

        measure = MeasureWidth(final_binary_raster, final_binary_transform, vector_path)
        measure.process(int=int, tol=tol, res=res)

        # Plot transects over the final binary raster
        fig, ax = plt.subplots(figsize=(10, 10))
        measure.display(clipped_data, ax=ax)

        #OUTPUT/EXPORTING OUTPUTS
        output_folder = input("Enter path to output folder: ").strip()
        
        dirname = os.path.dirname(raster_path)
        basename = os.path.basename(dirname)

        if not output_folder:
            print("No output folder provided. Exiting.")

            end_time = time.time()
            total_run_time = end_time - start_time

            return print(f"Total Run Time: {total_run_time:.2f} seconds")

        outpath = os.path.join(output_folder, basename, basename + "-road_final.tif")
        
        export(final_binary_raster, final_binary_transform, output_path=outpath)

        measure.export(os.path.join(output_folder, basename, basename + "-widths.geojson"))
        
        print("Width extraction completed.")
        print(measure.transects)

        print("Mean: ", measure.clipped_transects['width'].mean())

        measure.export(os.path.join(output_folder, basename, basename + "-vectorized_road.geojson"), gdf = measure.vectorized_roads)
        print("Exporting Vectorized road completed.")

        end_time = time.time()
        total_run_time = end_time - start_time
        print(f"Total Run Time: {total_run_time:.2f} seconds")

    else:
        print("Invalid option. Please enter T or E.")
        return #return to trk_ext input, and ask again

# Entry point of the script
if __name__ == "__main__":
    main()