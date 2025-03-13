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
        mapContainer: document.getElementById('map-container')
    };
    
    // State variables
    let state = {
        selectedLocation: null,
        currentImagePath: null,
        proportionsChart: null,
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
    
    function updateImageMode() {
        if(state.selectedLocation) {
            getSatelliteImage(state.selectedLocation.lat, state.selectedLocation.lng);
        }
    }
    
    function initMap() {
        // Create the map with custom options
        state.map = L.map('map-container', {
            zoomControl: false,
            attributionControl: true,
            doubleClickZoom: true,
            scrollWheelZoom: true
        }).setView([40, -95], 4);
        
        // Add zoom control to bottom right
        L.control.zoom({
            position: 'bottomright'
        }).addTo(state.map);
        
        // Add tile layer
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(state.map);
        
        // Add click event handler
        state.map.on('click', handleMapClick);
        
        // Add coordinate display on hover
        const coordInfo = L.control({position: 'bottomleft'});
        coordInfo.onAdd = function() {
            this._div = L.DomUtil.create('div', 'coord-info');
            this.update();
            return this._div;
        };
        coordInfo.update = function(coords) {
            this._div.innerHTML = coords ? 
                `<small>Lat: ${coords.lat.toFixed(4)}, Lng: ${coords.lng.toFixed(4)}</small>` : '';
        };
        coordInfo.addTo(state.map);
        
        state.map.on('mousemove', e => coordInfo.update(e.latlng));
        state.map.on('mouseout', () => coordInfo.update(null));
        
        // Set cursor to pointer over map
        elements.mapContainer.style.cursor = 'pointer';
    }
    
    function initNavbarScroll() {
        window.addEventListener('scroll', function() {
            const navbar = document.querySelector('.navbar');
            navbar.classList.toggle('navbar-scrolled', window.scrollY > 50);
        });
    }
    
    function handleMapClick(e) {
        const lat = e.latlng.lat.toFixed(6);
        const lng = e.latlng.lng.toFixed(6);
        
        setMapMarker(lat, lng);
        
        // Skip reverse geocoding and directly create a location with coordinates
        const locationName = `Location at ${lat}, ${lng}`;
        const location = {
            name: locationName,
            lat: lat,
            lng: lng
        };
        
        // Update the search input
        elements.locationSearch.value = locationName;
        
        // Store selected location
        state.selectedLocation = location;
        
        // Process the selected location
        handleSelectedLocation(location);
    }
    
    function setMapMarker(lat, lng) {
        // Remove existing marker if any
        if (state.marker) {
            state.map.removeLayer(state.marker);
        }
        
        // Create custom marker icon
        const customIcon = L.divIcon({
            className: 'custom-marker',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });
        
        // Create new marker with popup
        state.marker = L.marker([lat, lng], {
            icon: customIcon,
            draggable: true
        }).addTo(state.map);
        
        // Add popup to marker
        state.marker.bindPopup(`
            <div class="marker-popup">
                <strong>Selected Location</strong><br>
                <span class="text-muted">${lat}, ${lng}</span>
            </div>
        `).openPopup();
        
        // When marker is dragged, update coordinates
        state.marker.on('dragend', function(event) {
            const position = event.target.getLatLng();
            reverseGeocode(position.lat.toFixed(6), position.lng.toFixed(6));
        });
        
        // Smooth animation to marker location
        state.map.flyTo(
            [lat, lng], 
            state.map.getZoom() < 10 ? 10 : state.map.getZoom(), 
            { duration: 1, easeLinearity: 0.5 }
        );
    }
    
    function reverseGeocode(lat, lng) {
        showLoading('Looking up address...');
        
        fetch(`/reverse-geocode?lat=${lat}&lng=${lng}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                hideLoading();
                
                if (data.status === 'success') {
                    // Update the search input with the address
                    elements.locationSearch.value = data.address;
                    
                    // Create location object
                    const location = {
                        name: data.address,
                        lat: lat,
                        lng: lng
                    };
                    
                    // Store selected location
                    state.selectedLocation = location;
                    
                    // Highlight the search bar
                    elements.locationSearch.classList.add('border-primary');
                    setTimeout(() => {
                        elements.locationSearch.classList.remove('border-primary');
                    }, 1500);
                    
                    // Focus on search button
                    elements.searchButton.focus();
                } else {
                    handleGenericLocation(lat, lng);
                }
            })
            .catch((error) => {
                hideLoading();
                console.error('Error in reverse geocoding:', error);
                handleGenericLocation(lat, lng);
            });
    }
    
    function handleGenericLocation(lat, lng) {
        const locationName = `Location at ${lat}, ${lng}`;
        elements.locationSearch.value = locationName;
        state.selectedLocation = {
            name: locationName,
            lat: lat,
            lng: lng
        };
    }
    
    function searchLocation() {
        // If we already have a selectedLocation from map, use it directly
        if (state.selectedLocation && 
            elements.locationSearch.value === state.selectedLocation.name) {
            handleSelectedLocation(state.selectedLocation);
            return;
        }
        
        // Otherwise, perform text-based search
        const query = elements.locationSearch.value.trim();
        if (!query) return;
        
        showLoading('Searching for location...');
        elements.locationSuggestions.innerHTML = '';
        
        fetch(`/search-location?query=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.status === 'error' || data.suggestions.length === 0) {
                    elements.locationSuggestions.innerHTML = '<div class="list-group-item text-danger">No locations found. Try a different search.</div>';
                    return;
                }
                
                data.suggestions.forEach(suggestion => {
                    const item = document.createElement('button');
                    item.className = 'list-group-item list-group-item-action';
                    item.textContent = suggestion.name;
                    item.addEventListener('click', () => {
                        handleSelectedLocation(suggestion);
                        elements.locationSuggestions.innerHTML = '';
                    });
                    elements.locationSuggestions.appendChild(item);
                });
            })
            .catch(() => {
                hideLoading();
                elements.locationSuggestions.innerHTML = '<div class="list-group-item text-danger">Error searching location. Please try again.</div>';
            });
    }
    
    function handleSelectedLocation(location) {
        state.selectedLocation = location;
        elements.locationSearch.value = location.name;
        
        // Update location info
        elements.locationDisplay.textContent = location.name;
        elements.coordsDisplay.textContent = `Coordinates: ${location.lat}, ${location.lng}`;
        
        // Set map marker if it's not already set
        setMapMarker(location.lat, location.lng);
        
        // Show content section and get satellite image
        elements.contentSection.style.display = 'block';
        getSatelliteImage(location.lat, location.lng);
        
        // Scroll to content section
        scrollToContentSection();
    }
    
    function scrollToContentSection() {
        elements.contentSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function showLoading(message = 'Processing your request...') {
        document.getElementById('loading-message').textContent = message;
        state.loadingModal.show();
    }
    
    function hideLoading() {
        state.loadingModal.hide();
    }
    
    function getSatelliteImage(lat, lng) {
        elements.satelliteContainer.innerHTML = `
            <div class="placeholder-container">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted mt-3">Loading satellite imagery...</p>
            </div>`;
        elements.analyzeButton.disabled = true;
        
        const mode = document.querySelector('input[name="imageMode"]:checked').value;
        
        showLoading('Fetching satellite imagery...');
        
        fetch('/get-satellite-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lng, mode })
        })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.status === 'error') {
                    elements.satelliteContainer.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-exclamation-circle placeholder-icon text-danger"></i>
                            <p class="text-danger">${data.message}</p>
                        </div>`;
                    return;
                }
                
                if (!data.image_paths || data.image_paths.length === 0) {
                    elements.satelliteContainer.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-satellite-dish placeholder-icon text-danger"></i>
                            <p class="text-danger">No images available.</p>
                        </div>`;
                    return;
                }
                
                // Display images
                elements.satelliteContainer.innerHTML = '';
                
                if (mode === 'single' && data.image_paths.length > 0) {
                    displaySingleImage(data.image_paths[0]);
                } else if (mode === 'grid' && data.image_paths.length > 0) {
                    displayImageGrid(data.image_paths);
                } else {
                    elements.satelliteContainer.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-satellite-dish placeholder-icon text-danger"></i>
                            <p class="text-danger">No images available.</p>
                        </div>`;
                }
            })
            .catch(() => {
                hideLoading();
                elements.satelliteContainer.innerHTML = `
                    <div class="placeholder-container">
                        <i class="fas fa-exclamation-triangle placeholder-icon text-danger"></i>
                        <p class="text-danger">Error loading satellite imagery. Please try again.</p>
                    </div>`;
            });
    }
    
    function displaySingleImage(imagePath) {
        state.currentImagePath = imagePath;
        const img = document.createElement('img');
        
        loadImageWithRetry(img, state.currentImagePath);
        img.className = 'img-fluid satellite-image';
        img.alt = 'Satellite Image';
        
        const imageContainer = document.createElement('div');
        imageContainer.className = 'image-container';
        imageContainer.appendChild(img);
        
        // Add image info overlay
        imageContainer.innerHTML += `
            <div class="image-info-overlay">
                <div class="image-info-content">
                    <span><i class="fas fa-map-marker-alt"></i> ${state.selectedLocation.lat}, ${state.selectedLocation.lng}</span>
                </div>
            </div>`;
        
        elements.satelliteContainer.appendChild(imageContainer);
        elements.analyzeButton.disabled = false;
    }
    
    function displayImageGrid(imagePaths) {
        const gridContainer = document.createElement('div');
        gridContainer.className = 'satellite-grid';
        
        // Select the first image by default
        state.currentImagePath = imagePaths[0];
        
        imagePaths.forEach((path, index) => {
            const imgContainer = document.createElement('div');
            imgContainer.className = `satellite-grid-item ${index === 0 ? 'selected' : ''}`;
            
            const img = document.createElement('img');
            loadImageWithRetry(img, path);
            img.className = 'img-fluid satellite-grid-image';
            img.alt = `Satellite Image ${index + 1}`;
            
            const gridPosition = document.createElement('div');
            gridPosition.className = 'grid-position';
            gridPosition.innerHTML = `<span>${index + 1}</span>`;
            
            img.addEventListener('click', () => {
                document.querySelectorAll('.satellite-grid-item').forEach(item => {
                    item.classList.remove('selected');
                });
                imgContainer.classList.add('selected');
                state.currentImagePath = path;
                elements.analyzeButton.disabled = false;
            });
            
            imgContainer.appendChild(img);
            imgContainer.appendChild(gridPosition);
            gridContainer.appendChild(imgContainer);
        });
        
        elements.satelliteContainer.appendChild(gridContainer);
        
        // Add image info below the grid
        const gridInfo = document.createElement('div');
        gridInfo.className = 'grid-info mt-3';
        gridInfo.innerHTML = `
            <div class="d-flex align-items-center justify-content-center text-muted">
                <i class="fas fa-info-circle me-2"></i>
                <small>Click on an image to select it for analysis</small>
            </div>`;
        elements.satelliteContainer.appendChild(gridInfo);
        
        elements.analyzeButton.disabled = false;
    }
    
    function loadImageWithRetry(imgElement, src, maxRetries = 3) {
        let retries = 0;
        
        function tryLoad() {
            imgElement.src = `${src}?t=${new Date().getTime()}`;
            
            imgElement.onerror = function() {
                if (retries < maxRetries) {
                    retries++;
                    setTimeout(tryLoad, 1000 * Math.pow(2, retries - 1));
                } else {
                    imgElement.src = 'data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22200%22%20height%3D%22200%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20200%20200%22%20preserveAspectRatio%3D%22none%22%3E%3Cdefs%3E%3Cstyle%20type%3D%22text%2Fcss%22%3E%23holder_189b3ff33db%20text%20%7B%20fill%3A%23FF0000%3Bfont-weight%3Abold%3Bfont-family%3AArial%2C%20Helvetica%2C%20Open%20Sans%2C%20sans-serif%2C%20monospace%3Bfont-size%3A10pt%20%7D%20%3C%2Fstyle%3E%3C%2Fdefs%3E%3Cg%20id%3D%22holder_189b3ff33db%22%3E%3Crect%20width%3D%22200%22%20height%3D%22200%22%20fill%3D%22%23F5F5F5%22%3E%3C%2Frect%3E%3Cg%3E%3Ctext%20x%3D%2256.1953125%22%20y%3D%22104.5%22%3EImage%20not%20found%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E';
                }
            };
        }
        
        tryLoad();
    }
    
    function analyzeImage() {
        if (!state.currentImagePath) return;
        
        elements.maskContainer.innerHTML = `
            <div class="placeholder-container">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted mt-3">Generating segmentation mask...</p>
            </div>`;
        
        showLoading('Analyzing satellite imagery...');
        
        // Get segmentation mask
        fetch('/get-segmentation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: state.currentImagePath })
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    hideLoading();
                    elements.maskContainer.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-exclamation-circle placeholder-icon text-danger"></i>
                            <p class="text-danger">${data.message}</p>
                        </div>`;
                    return;
                }
                
                // Display the mask
                elements.maskContainer.innerHTML = '';
                const img = document.createElement('img');
                
                loadImageWithRetry(img, data.mask_path);
                img.className = 'img-fluid mask-image';
                img.alt = 'Segmentation Mask';
                
                const maskContainer = document.createElement('div');
                maskContainer.className = 'mask-container';
                maskContainer.appendChild(img);
                
                // Add legend for mask colors
                const maskLegend = document.createElement('div');
                maskLegend.className = 'mask-legend mt-3';
                maskLegend.innerHTML = `
                    <div class="legend-title mb-2">Color Legend</div>
                    <div class="legend-items">
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #1a9641;"></span>
                            <span class="legend-label">Vegetation</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #3288bd;"></span>
                            <span class="legend-label">Water</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #d73027;"></span>
                            <span class="legend-label">Buildings</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #f1b6da;"></span>
                            <span class="legend-label">Roads</span>
                        </div>
                    </div>`;
                
                elements.maskContainer.appendChild(maskContainer);
                elements.maskContainer.appendChild(maskLegend);
                
                // Show analysis section
                elements.analysisSection.style.display = 'flex';
                
                // Get land proportions
                getLandProportions(state.currentImagePath);
            })
            .catch(() => {
                hideLoading();
                elements.maskContainer.innerHTML = `
                    <div class="placeholder-container">
                        <i class="fas fa-exclamation-triangle placeholder-icon text-danger"></i>
                        <p class="text-danger">Error generating segmentation mask. Please try again.</p>
                    </div>`;
            });
    }
    
    function getLandProportions(imagePath) {
        fetch('/get-land-proportions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: imagePath })
        })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.status === 'success' && data.proportions) {
                    displayProportionsChart(data.proportions);
                }
            })
            .catch(() => {
                hideLoading();
            });
    }
    
    function displayProportionsChart(proportions) {
        // Destroy existing chart if it exists
        if (state.proportionsChart) {
            state.proportionsChart.destroy();
        }
        
        const ctx = document.getElementById('proportions-chart').getContext('2d');
        
        const labels = Object.keys(proportions);
        const values = Object.values(proportions);
        
        const backgroundColors = [
            '#1a9641', // Vegetation
            '#3288bd', // Water
            '#d73027', // Buildings
            '#f1b6da', // Roads
            '#fee08b', // Bareland
            '#66c2a5', // Other
        ];
        
        state.proportionsChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: backgroundColors.slice(0, labels.length),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                return `${label}: ${value.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    function identifySuitableLocations() {
        const purpose = elements.purposeSelect.value;
        if (purpose === 'Select purpose...') {
            alert('Please select a purpose for identification');
            return;
        }
        
        const minArea = elements.minAreaInput.value || 1000;
        
        elements.locationsResult.innerHTML = `
            <div class="placeholder-container">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted mt-3">Identifying suitable locations...</p>
            </div>`;
        
        showLoading('Identifying suitable locations...');
        
        fetch('/identify-suitable-locations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ purpose, min_area_sqm: minArea })
        })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.status === 'error') {
                    elements.locationsResult.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-exclamation-circle placeholder-icon text-danger"></i>
                            <p class="text-danger">${data.message}</p>
                        </div>`;
                    return;
                }
                
                if (!data.suitable_locations || data.suitable_locations.length === 0) {
                    elements.locationsResult.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-map-pin placeholder-icon text-muted"></i>
                            <p class="text-muted">No suitable locations found for this purpose.</p>
                        </div>`;
                    return;
                }
                
                // Display suitable locations
                displaySuitableLocations(data.suitable_locations, purpose);
            })
            .catch(() => {
                hideLoading();
                elements.locationsResult.innerHTML = `
                    <div class="placeholder-container">
                        <i class="fas fa-exclamation-triangle placeholder-icon text-danger"></i>
                        <p class="text-danger">Error identifying suitable locations. Please try again.</p>
                    </div>`;
            });
    }
    
    function displaySuitableLocations(locations, purpose) {
        elements.locationsResult.innerHTML = '';
        
        const resultsHeader = document.createElement('div');
        resultsHeader.className = 'results-header mb-3';
        resultsHeader.innerHTML = `
            <i class="fas fa-check-circle text-success"></i>
            <h6 class="mb-0">Found ${locations.length} suitable location${locations.length > 1 ? 's' : ''} for ${purpose.replace('_', ' ')}</h6>
        `;
        
        elements.locationsResult.appendChild(resultsHeader);
        
        const locationsContainer = document.createElement('div');
        locationsContainer.className = 'locations-container';
        
        locations.forEach((location, index) => {
            const card = document.createElement('div');
            card.className = 'location-card mb-3';
            
            card.innerHTML = `
                <div class="location-header">
                    <div class="location-number">${index + 1}</div>
                    <div class="location-title">Location ${String.fromCharCode(65 + index)}</div>
                </div>
                <div class="location-body">
                    <div class="location-stat">
                        <i class="fas fa-ruler-combined"></i>
                        <span>Area: ${location.area_sqm.toLocaleString()} sq.m</span>
                    </div>
                    <div class="location-stat">
                        <i class="fas fa-map-marker-alt"></i>
                        <span>Coordinates: ${location.lat}, ${location.lng}</span>
                    </div>
                    <div class="location-stat">
                        <i class="fas fa-percentage"></i>
                        <span>Suitability Score: ${location.suitability_score}%</span>
                    </div>
                    <button class="btn btn-sm btn-outline-primary mt-2 view-location-btn" 
                            data-lat="${location.lat}" 
                            data-lng="${location.lng}">
                        View on Map
                    </button>
                </div>
            `;
            
            const viewButton = card.querySelector('.view-location-btn');
            viewButton.addEventListener('click', function() {
                const lat = this.getAttribute('data-lat');
                const lng = this.getAttribute('data-lng');
                setMapMarker(lat, lng);
                state.map.setView([lat, lng], 15);
                elements.heroSection.scrollIntoView({ behavior: 'smooth' });
            });
            
            locationsContainer.appendChild(card);
        });
        
        elements.locationsResult.appendChild(locationsContainer);
    }
});