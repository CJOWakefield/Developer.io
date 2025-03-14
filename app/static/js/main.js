document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const elements = {
        locationSearch: document.getElementById('location-search'),
        searchButton: document.getElementById('search-button'),
        locationSuggestions: document.getElementById('location-suggestions'),
        satelliteContainer: document.getElementById('satellite-image-container'),
        maskContainer: document.getElementById('mask-image-container'),
        analyzeButton: document.getElementById('analyze-button'),
        imagerySection: document.getElementById('imagery-section'),
        analysisSection: document.getElementById('analysis-section'),
        identifyButton: document.getElementById('identify-button'),
        purposeSelect: document.getElementById('purpose-select'),
        minAreaInput: document.getElementById('min-area-input'),
        locationsResult: document.getElementById('suitable-locations-result'),
        singleMode: document.getElementById('singleMode'),
        gridMode: document.getElementById('gridMode'),
        contentSection: document.getElementById('content-section'),
        locationDisplay: document.getElementById('current-location-display'),
        coordsDisplay: document.getElementById('coordinates-display'),
        mapContainer: document.getElementById('map-container'),
        heroSection: document.getElementById('hero-section')
    };
    
    // State variables
    let state = {
        selectedLocation: null,
        currentImagePath: null,
        proportionsChart: null,
        derivedProportionsChart: null,
        map: null,
        marker: null,
        loadingModal: new bootstrap.Modal(document.getElementById('loadingModal'))
    };
    
    // Initialize app
    initMap();
    initNavbarScroll();
    
    // Event Listeners
    elements.searchButton.addEventListener('click', searchLocation);
    elements.locationSearch.addEventListener('keypress', e => { if(e.key === 'Enter') searchLocation(); });
    elements.analyzeButton.addEventListener('click', analyzeImage);
    elements.identifyButton.addEventListener('click', identifySuitableLocations);
    elements.singleMode.addEventListener('change', updateImageMode);
    elements.gridMode.addEventListener('change', updateImageMode);
    
    // Helper functions
    function showLoading(message = 'Processing your request...') {
        document.getElementById('loading-message').textContent = message;
        state.loadingModal.show();
    }
    
    function hideLoading() {
        state.loadingModal.hide();
    }
    
    function createPlaceholder(icon, message, isError = false) {
        return `
            <div class="placeholder-container">
                <i class="fas fa-${icon} placeholder-icon ${isError ? 'text-danger' : ''}"></i>
                <p class="text-muted">${message}</p>
            </div>
        `;
    }
    
    function initNavbarScroll() {
        window.addEventListener('scroll', function() {
            const navbar = document.querySelector('.navbar');
            if (window.scrollY > 50) {
                navbar.classList.add('navbar-scrolled');
            } else {
                navbar.classList.remove('navbar-scrolled');
            }
        });
    }
    
    function initMap() {
        // Initialize the map
        state.map = L.map(elements.mapContainer, {
            zoomControl: false,  // We'll add zoom control in a better position
            attributionControl: false  // Hide attribution for cleaner look
        }).setView([20, 0], 2);
        
        // Add zoom control to top-right
        L.control.zoom({
            position: 'topright'
        }).addTo(state.map);
        
        // Add attribution in bottom-right
        L.control.attribution({
            position: 'bottomright',
            prefix: false
        }).addAttribution('© OpenStreetMap contributors').addTo(state.map);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19
        }).addTo(state.map);
        
        // Add click event to map
        state.map.on('click', function(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            handleSelectedLocation(lat, lng);
        });
        
        // Resize map when window is resized
        window.addEventListener('resize', function() {
            state.map.invalidateSize();
        });
    }
    
    function searchLocation() {
        const query = elements.locationSearch.value.trim();
        
        if (!query) {
            return;
        }
        
        showLoading('Searching for location...');
        
        fetch(`/search-location?query=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.status === 'success' && data.suggestions && data.suggestions.length > 0) {
                    // If only one result, select it directly
                    if (data.suggestions.length === 1) {
                        const location = data.suggestions[0];
                        handleSelectedLocation(location.lat, location.lng);
                        elements.locationSuggestions.innerHTML = '';
                        return;
                    }
                    
                    // Otherwise show suggestions
                    displayLocationSuggestions(data.suggestions);
                } else {
                    elements.locationSuggestions.innerHTML = `
                        <div class="list-group-item list-group-item-action text-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            No locations found. Try a different search.
                        </div>
                    `;
                }
            })
            .catch(() => {
                hideLoading();
                elements.locationSuggestions.innerHTML = `
                    <div class="list-group-item list-group-item-action text-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error searching for location. Please try again.
                    </div>
                `;
            });
    }
    
    function displayLocationSuggestions(suggestions) {
        let html = '';
        
        suggestions.forEach(location => {
            html += `
                <button class="list-group-item list-group-item-action location-suggestion" 
                    data-lat="${location.lat}" 
                    data-lng="${location.lng}">
                    <i class="fas fa-map-marker-alt me-2 text-primary"></i>
                    ${location.name}
                </button>
            `;
        });
        
        elements.locationSuggestions.innerHTML = html;
        
        // Add event listeners to suggestions
        document.querySelectorAll('.location-suggestion').forEach(item => {
            item.addEventListener('click', function() {
                const lat = parseFloat(this.getAttribute('data-lat'));
                const lng = parseFloat(this.getAttribute('data-lng'));
                
                handleSelectedLocation(lat, lng);
                elements.locationSuggestions.innerHTML = '';
            });
        });
    }
    
    function handleSelectedLocation(lat, lng) {
        // Get the current image mode
        const mode = elements.gridMode.checked ? 'grid' : 'single';
        
        // Get satellite image
        getSatelliteImage(lat, lng, mode);
    }
    
    function updateImageMode() {
        const mode = elements.gridMode.checked ? 'grid' : 'single';
        
        if (state.selectedLocation) {
            showLoading('Loading ' + mode + ' view...');
            getSatelliteImage(state.selectedLocation.lat, state.selectedLocation.lng, mode);
        }
    }
    
    function getSatelliteImage(lat, lng, mode = 'single') {
        showLoading('Loading satellite imagery...');
        
        // Clear any existing images
        elements.satelliteContainer.innerHTML = '';
        
        // Use POST method instead of GET
        fetch('/get-satellite-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                lat: lat,
                lng: lng,
                mode: mode
            })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.text().then(text => {
                    try {
                        return JSON.parse(text);
                    } catch (e) {
                        console.error("Failed to parse JSON response:", text.substring(0, 100) + "...");
                        throw new Error("Invalid JSON response from server");
                    }
                });
            })
            .then(data => {
                hideLoading();
                
                if (data.status === 'success') {
                    // Store the current image path for analysis
                    if (mode === 'single' && data.image_path) {
                        state.currentImagePath = data.image_path;
                        
                        // Display the image
                        elements.satelliteContainer.innerHTML = `
                            <img src="${data.image_path}?t=${new Date().getTime()}" 
                                 alt="Satellite image" class="img-fluid satellite-image">
                        `;
                    } else if (data.image_paths && data.image_paths.length > 0) {
                        // For grid mode, display multiple images
                        state.currentImagePath = data.image_paths[0]; // Use the first image for analysis
                        
                        let gridHtml = '<div class="image-grid">';
                        data.image_paths.forEach(path => {
                            gridHtml += `
                                <div class="grid-item">
                                    <img src="${path}?t=${new Date().getTime()}" 
                                         alt="Satellite image" class="img-fluid satellite-image">
                                </div>
                            `;
                        });
                        gridHtml += '</div>';
                        elements.satelliteContainer.innerHTML = gridHtml;
                    } else {
                        throw new Error("No image paths in response");
                    }
                    
                    // Show the content section
                    elements.contentSection.style.display = 'block';
                    
                    // Scroll to content section
                    elements.contentSection.scrollIntoView({ behavior: 'smooth' });
                    
                    // Update location display
                    updateLocationDisplay(lat, lng);
                } else {
                    elements.satelliteContainer.innerHTML = createPlaceholder(
                        'exclamation-triangle',
                        data.message || 'Error loading satellite image',
                        true
                    );
                    console.error("Error in response:", data);
                }
            })
            .catch(error => {
                hideLoading();
                elements.satelliteContainer.innerHTML = createPlaceholder(
                    'exclamation-triangle',
                    'Error loading satellite image. Please try again.',
                    true
                );
                console.error("Fetch error:", error);
            });
    }
    
    function analyzeImage() {
        if (!state.currentImagePath) {
            alert('Please select a location first');
            return;
        }
        
        showLoading('Analyzing satellite imagery...');
        
        // Use the correct endpoint from app.py: /get-segmentation
        fetch('/get-segmentation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: state.currentImagePath })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                hideLoading();
                
                if (data.status === 'success') {
                    // Display the segmentation mask
                    elements.maskContainer.innerHTML = `
                        <img src="${data.mask_path}?t=${new Date().getTime()}" 
                             alt="Segmentation mask" class="mask-image" id="mask-image">
                    `;
                    
                    // Add hover functionality to the mask
                    const maskImage = document.getElementById('mask-image');
                    if (maskImage) {
                        addMaskHoverFunctionality(maskImage);
                    }
                    
                    // Now get the land proportions
                    getLandProportions(state.currentImagePath);
                } else {
                    elements.maskContainer.innerHTML = createPlaceholder(
                        'exclamation-triangle',
                        data.message || 'Error generating segmentation mask',
                        true
                    );
                }
            })
            .catch(error => {
                hideLoading();
                elements.maskContainer.innerHTML = createPlaceholder(
                    'exclamation-triangle',
                    'Error analyzing image. Please try again.',
                    true
                );
                console.error("Analysis error:", error);
            });
    }
    
    function getPixelColor(img, x, y) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        context.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
        
        try {
            const pixelData = context.getImageData(x, y, 1, 1).data;
            return {
                r: pixelData[0],
                g: pixelData[1],
                b: pixelData[2]
            };
        } catch (error) {
            console.error("Error getting pixel color:", error);
            return { r: 0, g: 0, b: 0 };
        }
    }
    
    function getLandTypeFromColor(color) {
        // Define color ranges for different land types
        const landTypes = [
            { name: 'Forest', color: { r: 0, g: 255, b: 0 } },
            { name: 'Water', color: { r: 0, g: 0, b: 255 } },
            { name: 'Urban', color: { r: 255, g: 255, b: 255 } },
            { name: 'Agriculture', color: { r: 255, g: 255, b: 0 } },
            { name: 'Barren', color: { r: 128, g: 128, b: 128 } },
            { name: 'Wetland', color: { r: 0, g: 255, b: 255 } },
            { name: 'Shrubland', color: { r: 255, g: 0, b: 255 } }
        ];
        
        // Find the closest color match
        let closestType = 'Unknown';
        let minDistance = Infinity;
        
        landTypes.forEach(type => {
            const distance = colorDistance(color, type.color);
            if (distance < minDistance) {
                minDistance = distance;
                closestType = type.name;
            }
        });
        
        return closestType;
    }
    
    function colorDistance(color1, color2) {
        // Calculate Euclidean distance between two colors
        return Math.sqrt(
            Math.pow(color1.r - color2.r, 2) +
            Math.pow(color1.g - color2.g, 2) +
            Math.pow(color1.b - color2.b, 2)
        );
    }
    
    function getLandProportions(imagePath) {
        fetch('/get-land-proportions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: imagePath })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success' && data.proportions) {
                    displayProportions(data.proportions);
                } else {
                    console.error("Error getting land proportions:", data);
                }
            })
            .catch(error => {
                console.error("Land proportions error:", error);
            });
    }
    
    function createProportionsChart(proportions) {
        const ctx = document.getElementById('proportions-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (state.proportionsChart) {
            state.proportionsChart.destroy();
        }
        
        // Prepare data for chart
        const labels = Object.keys(proportions);
        const data = Object.values(proportions);
        
        // Define colors for each land type
        const colors = {
            'Forest': 'rgba(0, 128, 0, 0.7)',
            'Water': 'rgba(0, 0, 255, 0.7)',
            'Urban': 'rgba(128, 128, 128, 0.7)',
            'Agriculture': 'rgba(255, 215, 0, 0.7)',
            'Barren': 'rgba(165, 42, 42, 0.7)',
            'Wetland': 'rgba(0, 255, 255, 0.7)',
            'Shrubland': 'rgba(255, 0, 255, 0.7)'
        };
        
        // Create chart
        state.proportionsChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: labels.map(label => colors[label] || 'rgba(128, 128, 128, 0.7)'),
                    borderColor: labels.map(label => colors[label]?.replace('0.7', '1') || 'rgba(128, 128, 128, 1)'),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                return `${label}: ${(value * 100).toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    function updateLocationDisplay(lat, lng) {
        // Update the location display with coordinates
        if (elements.locationDisplay) {
            elements.locationDisplay.textContent = 'Selected Location';
        }
        
        if (elements.coordsDisplay) {
            elements.coordsDisplay.textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        }
        
        // Try to get a readable address for this location
        fetch(`/reverse-geocode?lat=${lat}&lng=${lng}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.address) {
                    if (elements.locationDisplay) {
                        elements.locationDisplay.textContent = data.address;
                    }
                }
            })
            .catch(error => {
                console.error("Error in reverse geocoding:", error);
            });
    }
    
    function setMapMarker(lat, lng) {
        // If we already have a marker, update its position
        if (state.marker) {
            state.marker.setLatLng([lat, lng]);
        } 
        // Otherwise create a new marker
        else if (state.map) {
            state.marker = L.marker([lat, lng]).addTo(state.map);
        }
        
        // Center the map on the marker
        if (state.map) {
            state.map.setView([lat, lng], 15);
        }
    }
    
    function identifySuitableLocations() {
        const purpose = elements.purposeSelect.value;
        const minArea = elements.minAreaInput.value;
        
        if (purpose === 'Select purpose...') {
            alert('Please select a purpose');
            return;
        }
        
        if (!state.currentImagePath) {
            alert('Please analyze an image first');
            return;
        }
        
        showLoading('Identifying suitable locations...');
        
        fetch('/identify-locations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path: state.currentImagePath,
                purpose: purpose,
                min_area: minArea
            })
        })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.status === 'success') {
                    displaySuitableLocations(data.locations, purpose);
                } else {
                    elements.locationsResult.innerHTML = `
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            ${data.message || 'No suitable locations found.'}
                        </div>
                    `;
                }
            })
            .catch(() => {
                hideLoading();
                elements.locationsResult.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error identifying suitable locations. Please try again.
                    </div>
                `;
            });
    }
    
    function displaySuitableLocations(locations, purpose) {
        if (!locations || locations.length === 0) {
            elements.locationsResult.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    No suitable locations found for ${purpose.replace('_', ' ')}.
                </div>
            `;
            return;
        }
        
        let html = `
            <h5 class="mb-3">Suitable Locations for ${purpose.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h5>
            <div class="row">
        `;
        
        locations.forEach((location, index) => {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card h-100">
                        <div class="card-body">
                            <h6 class="card-title">Location ${index + 1}</h6>
                            <p class="card-text">
                                <strong>Area:</strong> ${location.area} sq.m<br>
                                <strong>Suitability:</strong> ${location.suitability}%
                            </p>
                            <button class="btn btn-sm btn-outline-primary view-location-btn" 
                                data-lat="${location.center[0]}" 
                                data-lng="${location.center[1]}">
                                <i class="fas fa-map-marker-alt me-1"></i> View on Map
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        
        elements.locationsResult.innerHTML = html;
        
        // Add event listeners to view location buttons
        document.querySelectorAll('.view-location-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const lat = parseFloat(this.getAttribute('data-lat'));
                const lng = parseFloat(this.getAttribute('data-lng'));
                
                if (state.map) {
                    state.map.setView([lat, lng], 18);
                    
                    // Add a special marker for this location
                    const locationMarker = L.marker([lat, lng], {
                        icon: L.divIcon({
                            className: 'location-marker',
                            html: '<i class="fas fa-star"></i>',
                            iconSize: [30, 30]
                        })
                    }).addTo(state.map);
                    
                    // Scroll to map
                    elements.heroSection.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
    }

    // Add the hover functionality for the mask
    function addMaskHoverFunctionality(maskImage) {
        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.className = 'mask-tooltip';
        tooltip.style.display = 'none';
        maskImage.parentNode.appendChild(tooltip);
        
        // Color mapping for land types
        const landTypeColors = {
            'urban': 'Urban Area',
            'agriculture': 'Agricultural Land',
            'forest': 'Forest',
            'water': 'Water Body',
            'barren': 'Barren Land'
        };
        
        // Add event listeners
        maskImage.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Position the tooltip
            tooltip.style.left = `${e.clientX}px`;
            tooltip.style.top = `${e.clientY - 30}px`;
            tooltip.style.display = 'block';
            
            // For now, just show a placeholder message
            // In a real implementation, we would need to get the pixel color
            // and map it to a land type
            tooltip.textContent = 'Hover over different areas to see land types';
        });
        
        maskImage.addEventListener('mouseleave', function() {
            tooltip.style.display = 'none';
        });
    }

    // Make sure these functions have access to the elements and state objects
    window.elements = elements;
    window.state = state;
    
    console.log("DOM fully loaded and event listeners attached");
});