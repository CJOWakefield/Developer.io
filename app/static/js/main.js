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
        // Add a small delay to ensure the modal has time to initialize before hiding
        setTimeout(() => {
            if (state.loadingModal) {
                state.loadingModal.hide();
                
                // Additional check to force-remove modal if it's still visible
                const modalElement = document.getElementById('loadingModal');
                if (modalElement && modalElement.classList.contains('show')) {
                    document.body.classList.remove('modal-open');
                    modalElement.classList.remove('show');
                    modalElement.style.display = 'none';
                    
                    // Remove modal backdrop if it exists
                    const backdrop = document.querySelector('.modal-backdrop');
                    if (backdrop) {
                        backdrop.remove();
                    }
                }
            }
        }, 100);
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
        }).addAttribution('© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors').addTo(state.map);
        
        // Add tile layer - using Stamen Terrain as a fallback
        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors',
            errorTileUrl: 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg'
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
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
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
            .catch(error => {
                hideLoading();
                console.error("Search error:", error);
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
        
        // Store the selected location
        state.selectedLocation = { lat, lng };
        
        // Reset analysis state when selecting a new location
        elements.maskContainer.innerHTML = createPlaceholder('fill-drip', 'Analyze an image to view segmentation mask');
        elements.analysisSection.style.display = 'none';
        elements.locationsResult.innerHTML = '';
        
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
        
        // Reset current image path
        state.currentImagePath = null;
        
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
                    
                    // Show the analysis section
                    elements.analysisSection.style.display = 'block';
                    
                    // Now get the land proportions
                    getLandProportions(state.currentImagePath);
                } else {
                    elements.maskContainer.innerHTML = createPlaceholder(
                        'exclamation-triangle',
                        data.message || 'Error generating segmentation mask',
                        true
                    );
                    hideLoading();
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
        // Set a timeout to ensure loading is hidden even if the request takes too long
        const loadingTimeout = setTimeout(() => {
            hideLoading();
            console.warn("Land proportions request timed out, but continuing...");
        }, 10000); // 10 seconds timeout
        
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
                clearTimeout(loadingTimeout);
                
                if (data.status === 'success' && data.proportions) {
                    displayProportions(data.proportions);
                } else {
                    console.error("Error getting land proportions:", data);
                    hideLoading();
                }
            })
            .catch(error => {
                clearTimeout(loadingTimeout);
                console.error("Land proportions error:", error);
                hideLoading();
            })
            .finally(() => {
                // Ensure loading is always hidden when the request completes
                hideLoading();
            });
    }
    
    function createProportionsChart(proportions) {
        const ctx = document.getElementById('proportions-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (state.proportionsChart) {
            state.proportionsChart.destroy();
        }
        
        // Filter to include only the specific segmentation mask classes
        const allowedClasses = ['agriculture', 'barren', 'forest', 'rangeland', 'unknown', 'urban', 'water'];
        const filteredProportions = {};
        
        // Convert to lowercase and filter
        Object.entries(proportions).forEach(([key, value]) => {
            const keyLower = key.toLowerCase();
            if (allowedClasses.includes(keyLower)) {
                filteredProportions[key] = value;
            }
        });
        
        console.log("Filtered proportions for land type distribution:", filteredProportions);
        
        // Prepare data for chart
        const labels = Object.keys(filteredProportions);
        const data = Object.values(filteredProportions);
        
        // Define colors for each land type - using more distinct colors
        const colors = {
            'agriculture': 'rgba(218, 165, 32, 0.8)', // Goldenrod
            'barren': 'rgba(160, 82, 45, 0.8)',      // Sienna
            'forest': 'rgba(34, 139, 34, 0.8)',      // Forest green
            'rangeland': 'rgba(107, 142, 35, 0.8)',  // Olive drab
            'unknown': 'rgba(128, 128, 128, 0.8)',   // Gray
            'urban': 'rgba(169, 169, 169, 0.8)',     // Dark gray
            'water': 'rgba(30, 144, 255, 0.8)',      // Dodger blue
        };
        
        // Create chart with improved colors
        state.proportionsChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: labels.map(label => {
                        const key = label.toLowerCase();
                        return colors[key] || 'rgba(128, 128, 128, 0.8)';
                    }),
                    borderColor: labels.map(label => {
                        const key = label.toLowerCase();
                        return colors[key]?.replace('0.8', '1') || 'rgba(128, 128, 128, 1)';
                    }),
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
                    },
                    title: {
                        display: true,
                        text: 'Land Type Distribution',
                        font: {
                            size: 16
                        }
                    }
                }
            }
        });
        
        // Create a second chart for urban density (vegetation vs developed)
        createUrbanDensityChart(proportions);
    }
    
    function createUrbanDensityChart(proportions) {
        try {
            const canvas = document.getElementById('derived-proportions-chart');
            if (!canvas) {
                console.error("Cannot find derived-proportions-chart canvas element");
                return;
            }
            
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                console.error("Cannot get 2d context from derived-proportions-chart canvas");
                return;
            }
            
            // Destroy previous chart if it exists
            if (state.derivedProportionsChart) {
                state.derivedProportionsChart.destroy();
            }
            
            // Calculate the three categories: Vegetated, Developed, and Unbuildable
            let vegetated = 0;
            let developed = 0;
            let unbuildable = 0;
            
            // Calculate categories from proportions
            Object.entries(proportions).forEach(([key, value]) => {
                const keyLower = key.toLowerCase();
                
                // Vegetated areas
                if (keyLower.includes('forest') || keyLower.includes('shrub') || 
                    keyLower.includes('agriculture') || keyLower.includes('vegetat') || 
                    keyLower.includes('rangeland')) {
                    vegetated += value;
                }
                // Developed areas
                else if (keyLower.includes('urban') || keyLower.includes('develop') || 
                         keyLower.includes('built')) {
                    developed += value;
                }
                // Unbuildable areas (water, wetlands, etc.)
                else if (keyLower.includes('water') || keyLower.includes('wetland') || 
                         keyLower.includes('barren')) {
                    unbuildable += value;
                }
                // Unknown areas - distribute proportionally or add to unbuildable
                else if (keyLower.includes('unknown')) {
                    unbuildable += value;
                }
            });
            
            // Create land use categories data
            const landUseCategories = {
                'Vegetated': vegetated,
                'Developed': developed,
                'Unbuildable': unbuildable
            };
            
            console.log("Land use categories:", landUseCategories);
            
            // Check if we have any data
            const totalValue = vegetated + developed + unbuildable;
            if (totalValue <= 0) {
                console.error("No valid data for land use categories chart");
                return;
            }
            
            // Normalize to ensure they sum to 100%
            const normalizedData = {
                'Vegetated': vegetated / totalValue,
                'Developed': developed / totalValue,
                'Unbuildable': unbuildable / totalValue
            };
            
            // Prepare data for chart
            const labels = Object.keys(normalizedData);
            const data = Object.values(normalizedData);
            
            // Define colors for land use categories
            const colors = {
                'Vegetated': 'rgba(76, 175, 80, 0.7)',    // Green
                'Developed': 'rgba(158, 158, 158, 0.7)',  // Gray
                'Unbuildable': 'rgba(33, 150, 243, 0.7)'  // Blue
            };
            
            // Create chart
            state.derivedProportionsChart = new Chart(ctx, {
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
                        },
                        title: {
                            display: true,
                            text: 'Land Use Categories',
                            font: {
                                size: 16
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error("Error creating land use categories chart:", error);
        }
    }
    
    function displayProportions(proportions) {
        try {
            createProportionsChart(proportions);
        } catch (error) {
            console.error("Error creating proportions chart:", error);
        } finally {
            // Always hide loading when done
            hideLoading();
        }
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
        // Create tooltip element if it doesn't exist
        let tooltip = document.querySelector('.mask-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.className = 'mask-tooltip';
            tooltip.style.display = 'none';
            document.body.appendChild(tooltip);
        }
        
        // Color mapping for land types
        const landTypeColors = {
            // RGB values for different land types
            '0,255,0': 'Forest',
            '0,0,255': 'Water',
            '255,255,255': 'Urban',
            '255,255,0': 'Agriculture',
            '128,128,128': 'Barren',
            '0,255,255': 'Wetland',
            '255,0,255': 'Shrubland',
            '107,142,35': 'Rangeland',
            '0,0,0': 'Unknown'
        };
        
        // Create an offscreen canvas for pixel analysis
        const offscreenCanvas = document.createElement('canvas');
        const offscreenCtx = offscreenCanvas.getContext('2d');
        
        // Wait for the image to load
        maskImage.onload = function() {
            offscreenCanvas.width = maskImage.naturalWidth;
            offscreenCanvas.height = maskImage.naturalHeight;
            offscreenCtx.drawImage(maskImage, 0, 0);
        };
        
        // Add event listeners
        maskImage.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * (this.naturalWidth / rect.width));
            const y = Math.floor((e.clientY - rect.top) * (this.naturalHeight / rect.height));
            
            try {
                // Get pixel color at cursor position
                const pixelData = offscreenCtx.getImageData(x, y, 1, 1).data;
                const colorKey = `${pixelData[0]},${pixelData[1]},${pixelData[2]}`;
                
                // Get land type from color
                let landType = landTypeColors[colorKey] || 'Unknown';
                
                // Position the tooltip
                tooltip.style.left = `${e.clientX + 10}px`;
                tooltip.style.top = `${e.clientY - 30}px`;
                tooltip.style.display = 'block';
                tooltip.textContent = landType;
                
                // Add color indicator to tooltip
                tooltip.style.borderLeft = `4px solid rgb(${colorKey})`;
            } catch (error) {
                console.error("Error getting pixel data:", error);
                tooltip.textContent = "Error reading land type";
            }
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