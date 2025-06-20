## main.py (updated version)
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                            QMessageBox, QInputDialog, QFileDialog, QSizePolicy, QPushButton)
from PyQt5.QtCore import Qt
import numpy as np
import os
from util import *  # Your existing utility classes
import time
import cv2
import math

class MplCanvas(FigureCanvas):
    """Custom FigureCanvas that handles both pyplot and Qt interactions"""
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def clear(self):
        self.ax.clear()
        self.draw()

class InteractiveCanvas(MplCanvas):
    """Special canvas for manual tracking with enhanced interaction"""
    def __init__(self, img, vector_path, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.img_orig = np.moveaxis(img, 0, -1)
        self.vector_data = gpd.read_file(vector_path).to_crs("EPSG:32651")
        self.line_points = []
        self.press_event = {'x': None, 'y': None}
        self.drag_threshold = 5
        self.measurement_complete = False
        
        # Setup plot
        self.ax.imshow(self.img_orig)
        self.ax.set_title("Left-click to draw, Right-click to undo, Middle-click to reset, Enter to finish.")
        self.ax.axis("off")
        
        # Connect events - need to store these ids for proper cleanup
        self._event_ids = [
            self.mpl_connect('button_press_event', self.onpress),
            self.mpl_connect('button_release_event', self.onrelease),
            self.mpl_connect('key_press_event', self.onkey),
            self.mpl_connect('scroll_event', self.on_scroll)  # Add scroll event handler
        ]

        # Enable zoom with mouse wheel
        self._zoom_handler = None
        self.enable_zoom()

    def enable_zoom(self):
        """Enable zoom/pan functionality"""
        if self._zoom_handler is None:
            self._zoom_handler = self.mpl_connect('scroll_event', self.on_scroll)

    def disable_zoom(self):
        """Disable zoom/pan functionality"""
        if self._zoom_handler is not None:
            self.mpl_disconnect(self._zoom_handler)
            self._zoom_handler = None

    def redraw(self):
        """Redraw the canvas while preserving zoom/pan state"""
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        self.ax.clear()
        self.ax.imshow(self.img_orig)
        self.ax.set_title("Left-click to draw, Right-click to undo, Middle-click to reset, Enter to finish.")
        self.ax.axis("off")
        
        if self.line_points:
            x_vals, y_vals = zip(*self.line_points)
            self.ax.plot(x_vals, y_vals, 'g-', linewidth=2)
            self.ax.plot(x_vals, y_vals, 'ro', markersize=4)
        
        # Restore view limits
        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)
        self.draw()

    def __del__(self):
        """Clean up event handlers"""
        for eid in self._event_ids:
            self.mpl_disconnect(eid)
        if self._zoom_handler is not None:
            self.mpl_disconnect(self._zoom_handler)

    def on_scroll(self, event):
        """Enhanced zoom with smooth scaling"""
        if event.inaxes != self.ax:
            return
            
        base_scale = 1.5  # More noticeable zoom
        zoom_factor = base_scale if event.button == 'up' else 1/base_scale
        
        # Get current limits and mouse position
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        
        # Calculate new bounds
        new_width = (xlim[1] - xlim[0]) * zoom_factor
        new_height = (ylim[1] - ylim[0]) * zoom_factor
        
        # Apply new bounds centered on mouse
        self.ax.set_xlim([xdata - new_width/2, xdata + new_width/2])
        self.ax.set_ylim([ydata - new_height/2, ydata + new_height/2])
        self.draw()

    def onpress(self, event):
        """Handle mouse press events"""
        if event.button == 1 and event.inaxes:  # Left click
            self.press_event['x'] = event.x
            self.press_event['y'] = event.y

    def onrelease(self, event):
        """Handle mouse release events"""
        if not event.inaxes:
            return

        if event.button == 1:  # Left click release
            dx = abs(event.x - self.press_event['x'])
            dy = abs(event.y - self.press_event['y'])
            if dx < self.drag_threshold and dy < self.drag_threshold:
                # Treat as a left-click (not drag)
                x, y = event.xdata, event.ydata
                self.line_points.append((x, y))
                self.redraw()

        elif event.button == 3:  # Right-click: delete last point
            if self.line_points:
                self.line_points.pop()
                self.redraw()

        elif event.button == 2:  # Middle-click: reset all
            self.line_points.clear()
            self.redraw()

    def onkey(self, event):
        """Handle key press events"""
        if event.key == 'enter' and len(self.line_points) >= 2 and not self.measurement_complete:
            x_vals, y_vals = zip(*self.line_points)
            total_length = sum(math.hypot(x1 - x0, y1 - y0)
                            for (x0, y0), (x1, y1) in zip(self.line_points[:-1], self.line_points[1:]))
            
            vector_length = self.vector_data.length.sum()
            
            mid_x = sum(x_vals) / len(x_vals)
            mid_y = sum(y_vals) / len(y_vals)
            
            # Add text annotation
            self.ax.text(mid_x, mid_y, f"Length: {total_length:.2f} meters",
                        color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            progress = (total_length / vector_length) * 100
            print(f"Progress: {progress:.2f}%")

            # Add legend
            self.ax.legend([f"FMR progress: {progress:.2f}%"], loc='lower center', 
                          fontsize=12, frameon=True)
            
            self.draw()
            self.measurement_complete = True
            
            # Show export prompt
            self.show_export_prompt()
            
        elif event.key == 'backspace' and not self.measurement_complete:
            if self.line_points:
                self.line_points.pop()
                self.redraw()

    def show_export_prompt(self):
        """Show a dialog asking if user wants to export"""
        # Use Qt's main thread to show the dialog
        self.parent_window.export_prompt()

class MainWindow(QMainWindow):
    """Main application window that hosts the workflow"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Initialize variables
        self.current_canvas = None
        self.toolbar = None
        
        # Start workflow
        self.start_workflow()
    
    def start_workflow(self):
        """Start the main workflow"""
        # Get user inputs
        trk_ext, ok = QInputDialog.getItem(self, "Workflow Selection", 
                                         "Do you want to track FMR progress or Extract information?", 
                                         ["Track (T)", "Extract (E)"], 0, False)
        if not ok:
            self.close()
            return
        
        trk_ext = trk_ext[0].upper()
        
        image_type, ok = QInputDialog.getItem(self, "Image Type", 
                                            "Enter the type of image:", 
                                            ["BSG", "PNEO"], 0, False)
        if not ok:
            self.close()
            return
        image_type = image_type.upper()
        
        # Get file paths
        raster_path, _ = QInputDialog.getText(self, "Raster Path", 
                                            "Enter the path to the raster:")
        if not raster_path:
            self.close()
            return
            
        vector_path, _ = QInputDialog.getText(self, "Vector Path", 
                                            "Enter the path to the vector (or press Enter to skip):")
        
        # Initialize preprocessing
        self.preprocessor = Preprocessing(raster_path, vector_path if vector_path else None)
        self.preprocessor.reproject()
        
        if trk_ext == "T":
            self.track_fmr_progress(vector_path)
        elif trk_ext == "E":
            self.extract_information(image_type)
        else:
            QMessageBox.warning(self, "Invalid Option", "Please enter T or E.")
            self.close()
    
    def track_fmr_progress(self, vector_path):
        """Handle FMR tracking workflow"""
        # Clip raster with buffer
        clipped_data, clipped_transform = self.preprocessor.clipraster(buffer_dist=25, bbox=True)
        
        # Show initial display
        self.show_preprocessed_data(clipped_data, clipped_transform)
        
        # Ask for tracking mode
        tracking_mode, ok = QInputDialog.getItem(self, "Tracking Mode", 
                                               "Do you want automatic or manual tracking?", 
                                               ["Automatic", "Manual"], 0, False)
        if not ok:
            return
            
        tracking_mode = tracking_mode[0].upper()
        
        if tracking_mode == "A":
            clipped_data, clipped_transform = self.preprocessor.clipraster(buffer_dist=25)
            self.automatic_tracking(clipped_data, clipped_transform)
        elif tracking_mode == "M":
            if not vector_path:
                QMessageBox.warning(self, "Error", "Vector path is required for manual tracking")
                return
            self.manual_tracking(clipped_data, vector_path)
        else:
            QMessageBox.warning(self, "Invalid Option", "Please enter A or M.")
    
    def automatic_tracking(self, clipped_data, clipped_transform):
        """Perform automatic FMR tracking with visualization before export"""
        try:
            # Apply filters
            filter = Filters(clipped_data)
            warm_raster = filter.enhance_image_warmth()
            stretch_raster = filter.enhance_linear_stretch()
            
            # Apply morphological operations
            morph = Morph()
            morph_warm = morph.process(warm_raster)
            morph_stretch = morph.process(stretch_raster)
            
            # Merge results
            merged_or = np.logical_or(morph_warm, morph_stretch)
            initial_binary_raster = merged_or
            
            # Final processing
            final_binary_raster = cv2.morphologyEx(
                initial_binary_raster.astype(np.uint8), 
                cv2.MORPH_CLOSE, 
                np.ones((3,3), np.uint8), 
                iterations=3
            )
            
            final_clipped_data, final_clipped_transform = self.preprocessor.clipraster(
                raster_data=final_binary_raster.astype(np.uint8),
                transform=clipped_transform,
                buffer_dist=1
            )
            
            final_clipped_data = np.squeeze(final_clipped_data)
            final_line = measure_line(final_clipped_data, final_clipped_transform, spacing=3)
            
            # Show results with option to proceed
            self.show_automatic_results(final_clipped_data, final_line, final_clipped_transform)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Automatic tracking failed: {str(e)}")

    def show_automatic_results(self, binary_raster, line_gdf, transform):
        """Show automatic tracking results with export option"""
        # Clear previous canvas
        self.clear_current_canvas()
        
        # Create results canvas
        self.current_canvas = MplCanvas(self)
        
        # Show binary raster
        show(binary_raster, ax=self.current_canvas.ax, transform=transform, cmap='gray')
        
        # Plot detected line
        line_gdf.plot(ax=self.current_canvas.ax, color='red', linewidth=2)
        
        # Add title and info
        length = line_gdf.length.values[0]
        self.current_canvas.ax.set_title(
            f"Automatic Tracking Results\nDetected Length: {length:.2f} meters", 
            fontsize=12
        )
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.current_canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.current_canvas)
        
        # Add export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(lambda: self.export_automatic_results(binary_raster, transform, line_gdf))
        self.layout.addWidget(export_btn)
        
        # Add continue button
        continue_btn = QPushButton("Continue Without Exporting")
        continue_btn.clicked.connect(self.close_results_view)
        self.layout.addWidget(continue_btn)

    def export_automatic_results(self, binary_raster, transform, line_gdf):
        """Handle export of automatic tracking results"""
        options = QFileDialog.Options()
        save_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select Output Directory", 
            "", 
            options=options
        )
        
        if save_dir:
            try:
                # Export binary raster
                raster_path = os.path.join(save_dir, "tracking_result.tif")
                export(binary_raster, transform, raster_path)
                
                # Export line
                line_path = os.path.join(save_dir, "detected_line.geojson")
                line_gdf.to_file(line_path, driver='GeoJSON')
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Results exported to:\n{raster_path}\n{line_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Export Error", 
                    f"Failed to export results: {str(e)}"
                )

    def close_results_view(self):
        """Close the results view and exit the application"""
        reply = QMessageBox.question(
            self,
            'Confirm Exit',
            'Are you sure you want to exit without exporting?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.close()  # This will trigger the closeEvent
        # If No is selected, the dialog closes but the application remains open

    def export_prompt(self):
        """Show export prompt after measurement is complete"""
        reply = QMessageBox.question(self, 'Export Drawing', 
                                   'Would you like to export the drawn line?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            self.export_drawn_line()
            self.current_canvas.close()

        else:
            # Reset measurement state and allow drawing again
            if hasattr(self.current_canvas, 'measurement_complete'):
                self.current_canvas.measurement_complete = False
                self.current_canvas.redraw()

    def export_drawn_line(self):
        """Handle export of the manually drawn line"""
        if not hasattr(self.current_canvas, 'line_points') or not self.current_canvas.line_points:
            QMessageBox.warning(self, "Error", "No line points to export")
            return
            
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Line", "", "Shapefiles (*.shp);;All Files (*)")
            
        if output_path:
            try:
                # Create LineString from points
                line_geom = [shapely.geometry.LineString(self.current_canvas.line_points)]
                gdf = gpd.GeoDataFrame(geometry=line_geom, crs="EPSG:32651")
                
                # Ensure proper file extension
                if not output_path.lower().endswith('.shp'):
                    output_path += '.shp'
                
                gdf.to_file(output_path)
                QMessageBox.information(self, "Success", f"Line saved to {output_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def manual_tracking(self, clipped_data, vector_path):
        """Handle manual FMR tracking with proper interaction"""
        # Clear previous canvas and toolbar
        self.clear_current_canvas()
        
        # Create interactive canvas
        self.current_canvas = InteractiveCanvas(clipped_data, vector_path, self)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.current_canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.current_canvas)
        
        # Set focus to canvas for key events
        self.current_canvas.setFocus()
    
    def clear_current_canvas(self):
        """Clear the current canvas and toolbar"""
        if self.toolbar:
            self.layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
            
        if self.current_canvas:
            self.layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None
    
    def extract_information(self, image_type):
        """Handle information extraction workflow"""
        if image_type == "PNEO":
            self.process_pneo()
        elif image_type == "BSG":
            self.process_bsg()
        else:
            QMessageBox.warning(self, "Invalid Image Type", "Please enter either BSG or PNEO.")
    
    def process_pneo(self):
        """Process PNEO images"""
        int, tol, res = 3, 0.15, 0.3 
        clipped_data, clipped_transform = self.preprocessor.clipraster(bbox=True)
        
        # Show original image if requested
        if self.ask_yes_no("View Image", "Do you want to view the image?"):
            self.show_stretched_image(clipped_data)
        
        # Apply filters
        filter = Filters(clipped_data)
        cielab = filter.cielab()
        
        self.morph = Morph()
        self.morph.threshold_cielab(cielab)
        initial_binary_raster = self.morph.thresholded
        
        self.final_processing(initial_binary_raster, clipped_data, clipped_transform, int, tol, res)
    
    def process_bsg(self):
        """Process BSG images"""
        int, tol, res = 3, 0.4, 0.3
        raster_data, _ = self.preprocessor.clipraster(bbox=True)
        clipped_data, clipped_transform = self.preprocessor.clipraster(buffer_dist=25)
        
        # Show original image if requested
        if self.ask_yes_no("View Image", "Do you want to view the image?"):
            self.show_stretched_image(clipped_data)
        
        # Apply filters
        filter = Filters(clipped_data)
        warm_raster = filter.enhance_image_warmth()
        stretch_raster = filter.enhance_linear_stretch()
        
        # Apply morphological operations
        self.morph = Morph()
        morph_warm = self.morph.process(warm_raster)
        morph_stretch = self.morph.process(stretch_raster)
        
        # Merge results
        merged_or = np.logical_or(morph_warm, morph_stretch)
        initial_binary_raster = merged_or
        
        self.final_processing(initial_binary_raster, clipped_data, clipped_transform, int, tol, res)
    
    def final_processing(self, initial_binary_raster, clipped_data, clipped_transform, int, tol, res):
        """Common final processing steps for both image types"""
        final_binary_transform = clipped_transform
        
        # Remove small islands and apply morphology
        final_binary_raster = self.morph.remove_small_islands(initial_binary_raster, min_size=1000)
        final_binary_raster = cv2.morphologyEx(final_binary_raster.astype(np.uint8), 
                                             cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)
        
        # Show comparison
        self.show_comparison(clipped_data, final_binary_raster)
        
        # Measure width
        measure = MeasureWidth(final_binary_raster, final_binary_transform, self.preprocessor.vector_path)
        measure.process(int=int, tol=tol, res=res)
        
        # Show results
        self.show_measurements(measure, clipped_data)
        
        # Export results
        self.export_results()
    
    def show_preprocessed_data(self, data, transform):
        """Display preprocessed data with vector overlay"""
        canvas = MplCanvas(self)
        show(data, ax=canvas.ax, transform=transform)
        if hasattr(self.preprocessor, 'vector_data'):
            self.preprocessor.vector_data.plot(ax=canvas.ax, color='red', linewidth=1, label='FMR')
            canvas.ax.legend(loc='upper right')
        canvas.ax.set_title("Preprocessed Data with Vector Overlay")
        canvas.ax.axis('off')
        self.show_canvas(canvas)
    
    def show_stretched_image(self, data):
        """Display stretched RGB image"""
        img = np.moveaxis(data, 0, -1)
        img = img[:, :, :3]
        img = np.stack([stretch_band(img[:, :, i]) for i in range(3)], axis=-1)
        
        canvas = MplCanvas(self)
        canvas.ax.imshow(img)
        canvas.ax.axis("off")
        canvas.ax.set_title("Clipped Image")
        self.show_canvas(canvas)
    
    def show_comparison(self, original, processed):
        """Show original and processed images side by side"""
        canvas = MplCanvas(self, width=12, height=6)
        
        # Original image
        canvas.ax[0].imshow(np.moveaxis(original, 0, -1))
        canvas.ax[0].axis("off")
        canvas.ax[0].set_title("Original Image")
        
        # Processed image
        canvas.ax[1].imshow(processed, cmap="gray")
        canvas.ax[1].axis("off")
        canvas.ax[1].set_title("Processed Image")
        
        plt.tight_layout()
        self.show_canvas(canvas)
    
    def show_measurements(self, measure, raster_data):
        """Display measurements on the raster"""
        canvas = MplCanvas(self)
        measure.display(raster_data, ax=canvas.ax)
        self.show_canvas(canvas)
    
    def show_results(self, data, line_gdf):
        """Display final results"""
        canvas = MplCanvas(self)
        show(data, ax=canvas.ax, cmap='gray')
        line_gdf.plot(ax=canvas.ax, color='red')
        canvas.ax.set_title("Results with Centerline")
        self.show_canvas(canvas)
    
    def show_canvas(self, canvas):
        """Display a canvas in the main window"""
        # Clear previous canvas
        if self.current_canvas:
            self.layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
        
        # Add new canvas
        if isinstance(canvas, FigureCanvas):
            self.current_canvas = canvas
        else:  # It's a matplotlib figure
            self.current_canvas = FigureCanvas(canvas)
        
        self.layout.addWidget(self.current_canvas)
        self.current_canvas.draw()
    
    def ask_yes_no(self, title, question):
        """Display a yes/no dialog"""
        reply = QMessageBox.question(self, title, question, 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes
    
    def export_results(self):
        """Handle export of results"""
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_folder:
            QMessageBox.information(self, "Info", "No output folder selected. Exiting.")
            return
        
        # Implement your export logic here
        # ...
        QMessageBox.information(self, "Success", "Export completed successfully.")
    
    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(
            self,
            'Confirm Exit',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clean up resources
            self.clear_current_canvas()
            event.accept()
        else:
            event.ignore()

def main():
    # Create application
    app = QApplication(sys.argv)
    
    # Set matplotlib to use Qt5 backend
    matplotlib.use('Qt5Agg')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()