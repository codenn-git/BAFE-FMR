# File monitoring system for automatic FMR database updates
import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import geopandas as gpd

# Add this import to your existing imports in fmr_gui-new.py
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

class FMRFileMonitor(FileSystemEventHandler):
    """File system event handler for monitoring shapefile and raster directories"""
    
    def __init__(self, shapefile_path, raster_folder, update_callback):
        self.shapefile_dir = os.path.dirname(shapefile_path)
        self.shapefile_path = shapefile_path
        self.raster_folder = raster_folder
        self.update_callback = update_callback
        self.last_update_time = time.time()
        self.update_delay = 5  # Wait 5 seconds before updating to avoid multiple rapid updates
        self.pending_update = False
        
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
            
        file_path = event.src_path
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check if it's a shapefile or raster image
        if self._is_relevant_file(file_path, file_ext):
            print(f"Detected new file: {file_path}")
            self._schedule_update()
    
    def on_moved(self, event):
        """Handle file move/rename events"""
        if event.is_directory:
            return
            
        dest_path = event.dest_path
        file_ext = os.path.splitext(dest_path)[1].lower()
        
        if self._is_relevant_file(dest_path, file_ext):
            print(f"Detected moved/renamed file: {dest_path}")
            self._schedule_update()
    
    def _is_relevant_file(self, file_path, file_ext):
        """Check if the file is relevant for FMR processing"""
        # Check if it's in the shapefile directory and is a shapefile
        if (file_path.startswith(self.shapefile_dir) and 
            file_ext == '.shp' and 
            file_path != self.shapefile_path):  # Don't trigger on master file
            return True
            
        # Check if it's in the raster folder and is a TIFF image
        if (file_path.startswith(self.raster_folder) and 
            file_ext == '.tif' and 
            'Tiff.tif' in os.path.basename(file_path)):  # Match your naming convention
            return True
            
        return False
    
    def _schedule_update(self):
        """Schedule a database update with a delay to avoid multiple rapid updates"""
        self.last_update_time = time.time()
        
        if not self.pending_update:
            self.pending_update = True
            # Use threading to avoid blocking the file watcher
            threading.Thread(target=self._delayed_update, daemon=True).start()
    
    def _delayed_update(self):
        """Execute the update after a delay"""
        time.sleep(self.update_delay)
        
        # Check if there have been more recent file changes
        if time.time() - self.last_update_time >= self.update_delay:
            try:
                print("Starting automatic FMR database update...")
                self.update_callback()
                print("Automatic FMR database update completed")
            except Exception as e:
                print(f"Error during automatic update: {e}")
            finally:
                self.pending_update = False


class AutoUpdater:
    """Main class to handle automatic FMR updates"""
    
    def __init__(self, shapefile_path, raster_folder, update_fmr_callback, update_db_callback):
        self.shapefile_path = shapefile_path
        self.raster_folder = raster_folder
        self.update_fmr_callback = update_fmr_callback  # updateFMRs function
        self.update_db_callback = update_db_callback    # getDatabase function
        self.observer = None
        self.is_monitoring = False
        
    def start_monitoring(self):
        """Start monitoring the directories for changes"""
        if self.is_monitoring:
            return
            
        try:
            # Create the file system event handler
            event_handler = FMRFileMonitor(
                self.shapefile_path, 
                self.raster_folder,
                self._handle_update
            )
            
            # Create and configure the observer
            self.observer = Observer()
            
            # Monitor shapefile directory
            shapefile_dir = os.path.dirname(self.shapefile_path)
            if os.path.exists(shapefile_dir):
                self.observer.schedule(event_handler, shapefile_dir, recursive=False)
                print(f"Monitoring shapefile directory: {shapefile_dir}")
            
            # Monitor raster directory
            if os.path.exists(self.raster_folder):
                self.observer.schedule(event_handler, self.raster_folder, recursive=True)
                print(f"Monitoring raster directory: {self.raster_folder}")
            
            # Start the observer
            self.observer.start()
            self.is_monitoring = True
            print("File monitoring started successfully")
            
        except Exception as e:
            print(f"Error starting file monitoring: {e}")
            self.is_monitoring = False
    
    def stop_monitoring(self):
        """Stop monitoring the directories"""
        if self.observer and self.is_monitoring:
            try:
                self.observer.stop()
                self.observer.join(timeout=5)
                self.is_monitoring = False
                print("File monitoring stopped")
            except Exception as e:
                print(f"Error stopping file monitoring: {e}")
    
    def _handle_update(self):
        """Handle the update process"""
        global gdf, filtered_gdf
        
        try:
            # First, update FMR shapefiles (merge new ones into master)
            self.update_fmr_callback(self.shapefile_path)
            
            # Reload the GeoDataFrame with updated data
            gdf = gpd.read_file(self.shapefile_path).to_crs(epsg=4326)
            filtered_gdf = gdf.copy()
            
            # Update the database with new image matches
            self.update_db_callback()
            
            # Recreate the map with updated data
            create_fmr_map()
            
            print("FMR data and database updated successfully")
            
        except Exception as e:
            print(f"Error during update process: {e}")
            raise