// JavaScript logic for FMR GUI
// July 3, 10:22PM fixed the display button

const selectedIds = new Set();
const geoLayers = {};
let currentMatchingImages = {};

function updateFMRList() {
    const ul = document.getElementById("fmr-list");
    ul.innerHTML = "";
    selectedIds.forEach(id => {
        const li = document.createElement("li");
        li.textContent = "FMR-" + id;
        ul.appendChild(li);
        const layer = geoLayers["geoLayer_" + id];
        if (layer) layer.setStyle({color: "red", weight: 3.5});
    });
    
    updateDisplayButtonState();
}

function selectFMR(fmr_id) {
    fetch("http://localhost:5000/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fmr_id: fmr_id })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "selected") {
            selectedIds.add(fmr_id);
            updateFMRList();

            // Fetch matching image paths
            return fetch("http://localhost:5000/get_matching_images", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fmr_id: fmr_id })
            });
        } else {
            alert("Already selected.");
            throw new Error("Already selected"); // Stop the chain
        }
    })
    .then(res => res.json())
    .then(imageData => {
        if (imageData.status === "success" && imageData.images.length > 0) {
            currentMatchingImages[fmr_id] = imageData.images;
            console.log(`FMR ${fmr_id} has ${imageData.images.length} images`);
        } else {
            currentMatchingImages[fmr_id] = []; // Explicitly mark as empty
            console.log(`FMR ${fmr_id} has no images`);
        }

        // Enable or disable display button based on available images
        updateDisplayButtonState();
    })
    .catch(err => {
        console.error("Error in selectFMR:", err);
        // If there was an error, make sure to update the button state anyway
        updateDisplayButtonState();
    });
}

function deselectFMR(fmr_id) {
    fetch("http://localhost:5000/deselect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({fmr_id: fmr_id})
    }).then(res => res.json())
    .then(data => {
        if (data.status === "deselected") {
            selectedIds.delete(fmr_id);

            delete currentMatchingImages[fmr_id];
            updateDisplayButtonState();
            
            updateFMRList();
            
            // Reset layer color to yellow
            const layer = geoLayers["geoLayer_" + fmr_id];
            if (layer) layer.setStyle({color: "yellow", weight: 3.5});
        } else {
            alert("FMR not selected or error occurred.");
        }
    });
}

function displaySelectedImages() {
    if (selectedIds.size === 0) {
        alert("No FMRs selected.");
        return;
    }

    // Remove previous overlays
    document.querySelectorAll(".leaflet-image-layer").forEach(el => el.remove());

    selectedIds.forEach(fmr_id => {
        const images = currentMatchingImages[fmr_id];
        if (!images || images.length === 0) return;

        const imagePath = images[0].path;
        fetch("http://localhost:5000/display_image", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fmr_id, image_path: imagePath })
        }).then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                const base64Image = data.image_data;
                const bounds = data.bounds;  // [[south, west], [north, east]]

                console.log("Adding overlay for FMR:", fmr_id, bounds);

                const overlay = L.imageOverlay(
                    `data:image/png;base64,${base64Image}`,
                    bounds,
                    { opacity: 1.0 }
                ).addTo(window._map);
            } else {
                console.error(`Image error: ${data.message}`);
            }
        });
    });
}

function runProcessing(fmr_id) {
    const images = currentMatchingImages[fmr_id];
    if (!images || images.length === 0) {
        alert("No matching images found for this FMR");
        return;
    }
    const trackRadio = document.getElementById(`track-${fmr_id}`);
    const extractRadio = document.getElementById(`extract-${fmr_id}`);

    let workflowType, workflowOptions = {};
    if (trackRadio.checked) {
        workflowType = 'track';
        workflowOptions.mode = document.getElementById(`auto-${fmr_id}`).checked ? 'automatic' : 'manual';
    } else if (extractRadio.checked) {
        workflowType = 'extract';
        workflowOptions.image_type = document.getElementById(`bsg-${fmr_id}`).checked ? 'BSG' : 'PNEO';
    } else {
        alert("Please select a processing workflow");
        return;
    }

    const imagePath = images[0].path;
    const resultsDiv = document.getElementById(`results-${fmr_id}`);
    resultsDiv.innerHTML = '<div>Processing... Please wait.</div>';
    resultsDiv.style.display = 'block';

    fetch("http://localhost:5000/process_fmr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            fmr_id: fmr_id,
            image_path: imagePath,
            workflow_type: workflowType,
            workflow_options: workflowOptions
        })
    }).then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            let resultHtml = `<div><b>Processing Results:</b><br>`;
            resultHtml += `<b>Status:</b> ${data.message}<br>`;
            if (data.length) resultHtml += `<b>Length:</b> ${data.length}<br>`;
            if (data.progress) resultHtml += `<b>Progress:</b> ${data.progress}<br>`;
            if (data.mean_width) resultHtml += `<b>Mean Width:</b> ${data.mean_width}<br>`;
            resultHtml += `</div>`;
            resultsDiv.innerHTML = resultHtml;
        } else {
            resultsDiv.innerHTML = `<div style="color: red;"><b>Error:</b> ${data.message}</div>`;
        }
    }).catch(err => {
        resultsDiv.innerHTML = `<div style="color: red;"><b>Error:</b> ${err.message}</div>`;
    });
}

function clearSelections() {
    fetch("http://localhost:5000/clear", {method: "POST"})
    .then(res => res.json())
    .then(data => {
        if (data.status === "cleared") {
            selectedIds.clear();
            
            Object.keys(currentMatchingImages).forEach(key => delete currentMatchingImages[key]);
            updateDisplayButtonState();

            updateFMRList();
            Object.values(geoLayers).forEach(layer => {
                layer.setStyle({color: "yellow", weight: 3.5});
            });
            document.querySelectorAll('[id^="processing-buttons-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="image-display-"]').forEach(el => el.style.display = 'none');
            document.querySelectorAll('[id^="results-"]').forEach(el => el.style.display = 'none');
        }
    });
}

function filterByProvince() {
    const province = document.getElementById("provinceSelect").value;
    fetch("http://localhost:5000/filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({province: province})
    }).then(res => res.json())
    .then(data => {
        if (data.status === "filtered") {
            alert(`Filtered to ${data.count} FMRs`);
            location.reload();
        }
    });
}

function downloadSelected() {
    if (selectedIds.size === 0) {
        alert("No FMRs selected");
        return;
    }
    fetch("http://localhost:5000/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            selected_ids: Array.from(selectedIds)
        })
    })
    .then(async response => {
        if (!response.ok) {
            let msg = "Download failed";
            try {
                const data = await response.json();
                if (data && data.message) msg = data.message;
            } catch (e) {}
            throw new Error(msg);
        }
        return response.blob();
    })
    .then(blob => {
        // Try to get filename from Content-Disposition header if possible
        let filename = "exported_fmr.zip";
        // The fetch API does not expose headers in .then(blob), so fallback to default name
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    })
    .catch(err => {
        alert(err.message);
    });
}

function updateFMRs() {
    if (confirm("Update FMR data? This may take a while.")) {
        fetch("http://localhost:5000/update_fmr", {method: "POST"})
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                alert("FMR data updated successfully!");
                location.reload();
            } else {
                alert(`Error updating FMR: ${data.message}`);
            }
        });
    }
}

function updateDisplayButtonState() {
    const btn = document.getElementById("displayImagesBtn");
    
    if (!btn) {
        console.error("Display button not found!");
        return;
    }

    const hasImages = Array.from(selectedIds).some(fmr_id =>
        currentMatchingImages[fmr_id] && currentMatchingImages[fmr_id].length > 0
    );

    console.log("Updating display button state:", {
        selectedIds: Array.from(selectedIds),
        hasImages: hasImages,
        currentMatchingImages: currentMatchingImages
    });

    btn.disabled = !hasImages;
    
    // Update button text to provide feedback
    if (selectedIds.size === 0) {
        btn.textContent = "Display Images";
    } else if (hasImages) {
        btn.textContent = "Display Images";
    } else {
        btn.textContent = "No Images Available";
    }
}