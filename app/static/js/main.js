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
        heroSection: document.getElementById('hero-section'),
        loadingMessage: document.getElementById('loading-message')
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
    console.log('DeveloperIO Satellite Imagery Analysis App initialized');
    
    // Map initialization
    function initMap() {
        state.map = L.map('map-container').setView([20, 0], 2);
        
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        }).addTo(state.map);
        
        state.map.on('click', function(e) {
            const lat = e.latlng.lat.toFixed(6);
            const lng = e.latlng.lng.toFixed(6);
            
            setMapMarker(lat, lng);
            selectLocation({
                name: `Location at ${lat}, ${lng}`,
                lat: lat,
                lng: lng
            });
            scrollToContentSection();
        });
    }
    
    // Set map marker
    function setMapMarker(lat, lng) {
        if (state.marker) state.map.removeLayer(state.marker);
        
        const customIcon = L.divIcon({
            className: 'custom-marker',
            iconSize: [16, 16]
        });
        
        state.marker = L.marker([lat, lng], {icon: customIcon}).addTo(state.map);
        state.map.setView([lat, lng], 15);
    }
    
    // Initialize navbar scroll behavior
    function initNavbarScroll() {
        window.addEventListener('scroll', function() {
            const navbar = document.querySelector('.navbar');
            navbar.classList.toggle('navbar-scrolled', window.scrollY > 50);
        });
    }
    
    // Scroll to content section
    function scrollToContentSection() {
        elements.contentSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Loading state management
    function showLoading(message = 'Processing your request...') {
        elements.loadingMessage.textContent = message;
        state.loadingModal.show();
    }
    
    function hideLoading() {
        state.loadingModal.hide();
    }
    
    // Image loading with retry
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
    
    // Event Listeners
    elements.searchButton.addEventListener('click', searchLocation);
    elements.locationSearch.addEventListener('keypress', e => { if(e.key === 'Enter') searchLocation(); });
    elements.analyzeButton.addEventListener('click', analyzeImage);
    elements.identifyButton.addEventListener('click', identifySuitableLocations);
    elements.singleMode.addEventListener('change', () => { if(state.selectedLocation) getSatelliteImage(state.selectedLocation.lat, state.selectedLocation.lng); });
    elements.gridMode.addEventListener('change', () => { if(state.selectedLocation) getSatelliteImage(state.selectedLocation.lat, state.selectedLocation.lng); });
    
    // Search location function
    function searchLocation() {
        const query = elements.locationSearch.value.trim();
        if (!query) return;
        
        console.log('Searching location:', query);
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
                        selectLocation(suggestion);
                        elements.locationSuggestions.innerHTML = '';
                        setMapMarker(suggestion.lat, suggestion.lng);
                        scrollToContentSection();
                    });
                    elements.locationSuggestions.appendChild(item);
                });
            })
            .catch(error => {
                console.error('Error searching location:', error);
                hideLoading();
                elements.locationSuggestions.innerHTML = '<div class="list-group-item text-danger">Error searching location. Please try again.</div>';
            });
    }
    
    // Select location function
    function selectLocation(location) {
        state.selectedLocation = location;
        elements.locationSearch.value = location.name;
        
        // Update location info
        elements.locationDisplay.textContent = location.name;
        elements.coordsDisplay.textContent = `Coordinates: ${location.lat}, ${location.lng}`;
        
        // Show content section and get satellite image
        elements.contentSection.style.display = 'block';
        getSatelliteImage(location.lat, location.lng);
    }
    
    // Get satellite image function
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
        
        console.log(`Requesting satellite image for: ${lat}, ${lng}, mode: ${mode}`);
        showLoading('Fetching satellite imagery...');
        
        fetch('/get-satellite-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lng, mode })
        })
            .then(response => response.json())
            .then(data => {
                console.log('Satellite image response:', data);
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
                    // Display single image
                    state.currentImagePath = data.image_paths[0];
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
                } else if (mode === 'grid' && data.image_paths.length > 0) {
                    // Display grid of images
                    const gridContainer = document.createElement('div');
                    gridContainer.className = 'satellite-grid';
                    
                    // Select the first image by default
                    state.currentImagePath = data.image_paths[0];
                    
                    data.image_paths.forEach((path, index) => {
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
                } else {
                    elements.satelliteContainer.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-satellite-dish placeholder-icon text-danger"></i>
                            <p class="text-danger">No images available.</p>
                        </div>`;
                }
            })
            .catch(error => {
                console.error('Error getting satellite image:', error);
                hideLoading();
                elements.satelliteContainer.innerHTML = `
                    <div class="placeholder-container">
                        <i class="fas fa-exclamation-triangle placeholder-icon text-danger"></i>
                        <p class="text-danger">Error loading satellite imagery. Please try again.</p>
                    </div>`;
            });
    }
    
    // Analyze image function
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
            .catch(error => {
                console.error('Error getting segmentation mask:', error);
                hideLoading();
                elements.maskContainer.innerHTML = `
                    <div class="placeholder-container">
                        <i class="fas fa-exclamation-triangle placeholder-icon text-danger"></i>
                        <p class="text-danger">Error generating segmentation mask. Please try again.</p>
                    </div>`;
            });
    }
    
    // Get land proportions function
    function getLandProportions(imagePath) {
        fetch('/get-land-proportions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: imagePath })
        })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.status === 'error') {
                    console.error('Error getting land proportions:', data.message);
                    return;
                }
                
                // Create/update chart
                createProportionsChart(data.proportions);
            })
            .catch(error => {
                console.error('Error getting land proportions:', error);
                hideLoading();
            });
    }
    
    // Create proportions chart function
    function createProportionsChart(proportions) {
        const ctx = document.getElementById('proportions-chart').getContext('2d');
        
        // Destroy previous chart if exists
        if (state.proportionsChart) {
            state.proportionsChart.destroy();
        }
        
        // Extract and format data
        const labels = Object.keys(proportions);
        const data = Object.values(proportions);
        const colors = generateColors(labels.length);
        
        // Format labels to be more readable
        const formattedLabels = labels.map(label => 
            label.split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ')
        );
        
        state.proportionsChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: formattedLabels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 1,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            font: {
                                family: "'Circular', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', sans-serif",
                                size: 12
                            },
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        titleColor: '#222222',
                        bodyColor: '#222222',
                        borderColor: '#dddddd',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 12,
                        boxPadding: 6,
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.formattedValue}%`;
                            }
                        }
                    }
                },
                animation: {
                    animateScale: true,
                    animateRotate: true
                }
            }
        });
    }
    
    // Generate colors function
    function generateColors(count) {
        const colors = [
            '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56', '#9966FF',
            '#FF9F40', '#C9CBCF', '#7BC8A4', '#E7E9ED', '#1D2C4D'
        ];
        
        if (count > colors.length) {
            for (let i = colors.length; i < count; i++) {
                const r = Math.floor(Math.random() * 255);
                const g = Math.floor(Math.random() * 255);
                const b = Math.floor(Math.random() * 255);
                colors.push(`rgb(${r},${g},${b})`);
            }
        }
        
        return colors.slice(0, count);
    }
    
    // Identify suitable locations function
    function identifySuitableLocations() {
        const purpose = elements.purposeSelect.value;
        const minArea = parseFloat(elements.minAreaInput.value);
        
        if (purpose === 'Select purpose...' || isNaN(minArea)) {
            elements.locationsResult.innerHTML = `
                <div class="placeholder-container">
                    <i class="fas fa-exclamation-circle placeholder-icon text-danger"></i>
                    <p class="text-danger">Please select a valid purpose and minimum area.</p>
                </div>`;
            return;
        }
        
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
            body: JSON.stringify({
                purpose: purpose,
                min_area_sqm: minArea
            })
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
                
                // Display suitable locations
                if (data.suitable_locations && data.suitable_locations.length > 0) {
                    const locationsHTML = data.suitable_locations.map((location, index) => `
                        <div class="location-card mb-3">
                            <div class="location-header">
                                <div class="location-number">${index + 1}</div>
                                <h6 class="mb-0">Suitable Location</h6>
                            </div>
                            <div class="location-body">
                                <div class="location-stat">
                                    <i class="fas fa-ruler-combined"></i>
                                    <span>Area: ${location.area_sqm.toFixed(2)} sq.m</span>
                                </div>
                                <div class="location-stat">
                                    <i class="fas fa-map-marker-alt"></i>
                                    <span>Coordinates: ${location.center.lat.toFixed(6)}, ${location.center.lon.toFixed(6)}</span>
                                </div>
                                <button class="btn btn-sm btn-outline-primary mt-2 view-on-map-btn" 
                                    data-lat="${location.center.lat}" 
                                    data-lon="${location.center.lon}">
                                    <i class="fas fa-map"></i> View on Map
                                </button>
                            </div>
                        </div>`
                    ).join('');
                    
                    elements.locationsResult.innerHTML = `
                        <div class="results-header">
                            <i class="fas fa-check-circle text-success"></i>
                            <h6 class="mb-0">Found ${data.suitable_locations.length} suitable location(s)</h6>
                        </div>
                        <div class="locations-container mt-3">
                            ${locationsHTML}
                        </div>`;
                    
                    // Add event listeners to "View on Map" buttons
                    document.querySelectorAll('.view-on-map-btn').forEach(btn => {
                        btn.addEventListener('click', function() {
                            const lat = this.getAttribute('data-lat');
                            const lon = this.getAttribute('data-lon');
                            
                            // Scroll back to map and set marker
                            elements.heroSection.scrollIntoView({ behavior: 'smooth' });
                            setTimeout(() => setMapMarker(lat, lon), 500);
                        });
                    });
                } else {
                    elements.locationsResult.innerHTML = `
                        <div class="placeholder-container">
                            <i class="fas fa-info-circle placeholder-icon text-info"></i>
                            <p class="text-muted">No suitable locations found. Try adjusting your criteria.</p>
                        </div>`;
                }
            })
            .catch(error => {
                console.error('Error identifying suitable locations:', error);
                hideLoading();
                elements.locationsResult.innerHTML = `
                    <div class="placeholder-container">
                        <i class="fas fa-exclamation-triangle placeholder-icon text-danger"></i>
                        <p class="text-danger">Error identifying suitable locations. Please try again.</p>
                    </div>`;
            });
    }
    
    // Initialize focus
    elements.locationSearch.focus();
});