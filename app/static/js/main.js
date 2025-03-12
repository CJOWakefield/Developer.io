document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const locationSearchInput = document.getElementById('location-search');
    const searchButton = document.getElementById('search-button');
    const locationSuggestions = document.getElementById('location-suggestions');
    const satelliteImageContainer = document.getElementById('satellite-image-container');
    const maskImageContainer = document.getElementById('mask-image-container');
    const analyzeButton = document.getElementById('analyze-button');
    const imagerySection = document.getElementById('imagery-section');
    const analysisSection = document.getElementById('analysis-section');
    const identifyButton = document.getElementById('identify-button');
    const purposeSelect = document.getElementById('purpose-select');
    const minAreaInput = document.getElementById('min-area-input');
    const suitableLocationsResult = document.getElementById('suitable-locations-result');
    const singleModeRadio = document.getElementById('singleMode');
    const gridModeRadio = document.getElementById('gridMode');
    
    // State variables
    let selectedLocation = null;
    let currentImagePath = null;
    let proportionsChart = null;
    
    // Display debug information in console
    console.log('Satellite Imagery Analysis App initialized');
    
    // Function to handle image loading failures with retry
    function loadImageWithRetry(imgElement, src, maxRetries = 3) {
        let retries = 0;
        
        function tryLoad() {
            // Add a timestamp to prevent caching
            const timestamp = new Date().getTime();
            imgElement.src = `${src}?t=${timestamp}`;
            
            imgElement.onerror = function() {
                if (retries < maxRetries) {
                    console.log(`Retrying image load (${retries + 1}/${maxRetries}): ${src}`);
                    retries++;
                    // Wait before retrying with exponential backoff
                    setTimeout(tryLoad, 1000 * Math.pow(2, retries - 1));
                } else {
                    console.error(`Failed to load image after ${maxRetries} attempts:`, src);
                    imgElement.src = 'data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22200%22%20height%3D%22200%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20200%20200%22%20preserveAspectRatio%3D%22none%22%3E%3Cdefs%3E%3Cstyle%20type%3D%22text%2Fcss%22%3E%23holder_189b3ff33db%20text%20%7B%20fill%3A%23FF0000%3Bfont-weight%3Abold%3Bfont-family%3AArial%2C%20Helvetica%2C%20Open%20Sans%2C%20sans-serif%2C%20monospace%3Bfont-size%3A10pt%20%7D%20%3C%2Fstyle%3E%3C%2Fdefs%3E%3Cg%20id%3D%22holder_189b3ff33db%22%3E%3Crect%20width%3D%22200%22%20height%3D%22200%22%20fill%3D%22%23F5F5F5%22%3E%3C%2Frect%3E%3Cg%3E%3Ctext%20x%3D%2256.1953125%22%20y%3D%22104.5%22%3EImage%20not%20found%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E';
                }
            };
        }
        
        tryLoad();
    }
    
    // Event Listeners
    searchButton.addEventListener('click', searchLocation);
    locationSearchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchLocation();
        }
    });
    
    analyzeButton.addEventListener('click', analyzeImage);
    identifyButton.addEventListener('click', identifySuitableLocations);
    
    // Search location function
    function searchLocation() {
        const query = locationSearchInput.value.trim();
        if (!query) return;
        
        console.log('Searching location:', query);
        locationSuggestions.innerHTML = '<p class="text-center"><small>Searching...</small></p>';
        
        fetch(`/search-location?query=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                locationSuggestions.innerHTML = '';
                
                if (data.status === 'error' || data.suggestions.length === 0) {
                    locationSuggestions.innerHTML = `<p class="text-danger">No locations found. Try a different search.</p>`;
                    return;
                }
                
                data.suggestions.forEach(suggestion => {
                    const item = document.createElement('button');
                    item.className = 'list-group-item list-group-item-action';
                    item.textContent = suggestion.name;
                    item.addEventListener('click', () => {
                        selectLocation(suggestion);
                    });
                    locationSuggestions.appendChild(item);
                });
            })
            .catch(error => {
                console.error('Error searching location:', error);
                locationSuggestions.innerHTML = `<p class="text-danger">Error searching location. Please try again.</p>`;
            });
    }
    
    // Select location function
    function selectLocation(location) {
        selectedLocation = location;
        locationSuggestions.innerHTML = '';
        locationSearchInput.value = location.name;
        
        // Show imagery section
        imagerySection.style.display = 'flex';
        
        // Get satellite image
        getSatelliteImage(location.lat, location.lng);
    }
    
    // Get satellite image function
    function getSatelliteImage(lat, lng) {
        satelliteImageContainer.innerHTML = '<p class="text-center">Loading satellite imagery...</p>';
        analyzeButton.disabled = true;
        
        // Determine mode
        const mode = document.querySelector('input[name="imageMode"]:checked').value;
        
        console.log(`Requesting satellite image for: ${lat}, ${lng}, mode: ${mode}`);
        
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
            .then(response => response.json())
            .then(data => {
                console.log('Satellite image response:', data);
                
                if (data.status === 'error') {
                    satelliteImageContainer.innerHTML = `<p class="text-danger">${data.message}</p>`;
                    return;
                }
                
                // Check if we have images
                if (!data.image_paths || data.image_paths.length === 0) {
                    satelliteImageContainer.innerHTML = '<p class="text-danger">No images available.</p>';
                    return;
                }
                
                // Display the images
                satelliteImageContainer.innerHTML = '';
                
                if (mode === 'single' && data.image_paths.length > 0) {
                    // Display single image
                    currentImagePath = data.image_paths[0];
                    const img = document.createElement('img');
                    
                    // Debug information
                    console.log('Loading single image:', currentImagePath);
                    
                    // Use loadImageWithRetry for better error handling
                    loadImageWithRetry(img, currentImagePath);
                    img.className = 'img-fluid satellite-image';
                    img.alt = 'Satellite Image';
                    
                    satelliteImageContainer.appendChild(img);
                    analyzeButton.disabled = false;
                } else if (mode === 'grid' && data.image_paths.length > 0) {
                    // Display grid of images
                    const gridContainer = document.createElement('div');
                    gridContainer.className = 'satellite-grid';
                    
                    // Debug information
                    console.log('Loading grid images, count:', data.image_paths.length);
                    
                    // Select the first image by default
                    currentImagePath = data.image_paths[0];
                    
                    data.image_paths.forEach((path, index) => {
                        const imgContainer = document.createElement('div');
                        imgContainer.className = 'satellite-grid-item';
                        if (index === 0) {
                            imgContainer.classList.add('selected');
                        }
                        
                        const img = document.createElement('img');
                        // Use loadImageWithRetry for better error handling
                        loadImageWithRetry(img, path);
                        img.className = 'img-fluid satellite-grid-image';
                        img.alt = `Satellite Image ${index + 1}`;
                        
                        img.addEventListener('click', () => {
                            // Highlight selected image
                            document.querySelectorAll('.satellite-grid-item').forEach(item => {
                                item.classList.remove('selected');
                            });
                            imgContainer.classList.add('selected');
                            currentImagePath = path;
                            analyzeButton.disabled = false;
                        });
                        
                        imgContainer.appendChild(img);
                        gridContainer.appendChild(imgContainer);
                    });
                    
                    satelliteImageContainer.appendChild(gridContainer);
                    analyzeButton.disabled = false;
                } else {
                    satelliteImageContainer.innerHTML = '<p class="text-danger">No images available.</p>';
                }
            })
            .catch(error => {
                console.error('Error getting satellite image:', error);
                satelliteImageContainer.innerHTML = `<p class="text-danger">Error loading satellite imagery. Please try again.</p>`;
            });
    }
    
    // Analyze image function
    function analyzeImage() {
        if (!currentImagePath) return;
        
        maskImageContainer.innerHTML = '<p class="text-center">Generating segmentation mask...</p>';
        
        // First, get segmentation mask
        fetch('/get-segmentation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_path: currentImagePath
            })
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    maskImageContainer.innerHTML = `<p class="text-danger">${data.message}</p>`;
                    return;
                }
                
                // Display the mask
                maskImageContainer.innerHTML = '';
                const img = document.createElement('img');
                
                // Debug information
                console.log('Loading mask image:', data.mask_path);
                
                // Use loadImageWithRetry for better error handling
                loadImageWithRetry(img, data.mask_path);
                img.className = 'img-fluid mask-image';
                img.alt = 'Segmentation Mask';
                
                maskImageContainer.appendChild(img);
                
                // Show analysis section
                analysisSection.style.display = 'flex';
                
                // Get land proportions
                getLandProportions(currentImagePath);
            })
            .catch(error => {
                console.error('Error getting segmentation mask:', error);
                maskImageContainer.innerHTML = `<p class="text-danger">Error generating segmentation mask. Please try again.</p>`;
            });
    }
    
    // Get land proportions function
    function getLandProportions(imagePath) {
        fetch('/get-land-proportions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_path: imagePath
            })
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    console.error('Error getting land proportions:', data.message);
                    return;
                }
                
                // Create/update chart
                createProportionsChart(data.proportions);
            })
            .catch(error => {
                console.error('Error getting land proportions:', error);
            });
    }
    
    // Create proportions chart function
    function createProportionsChart(proportions) {
        const ctx = document.getElementById('proportions-chart').getContext('2d');
        
        // Destroy previous chart if exists
        if (proportionsChart) {
            proportionsChart.destroy();
        }
        
        // Extract labels and data
        const labels = Object.keys(proportions);
        const data = Object.values(proportions);
        
        // Generate colors
        const colors = generateColors(labels.length);
        
        proportionsChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.formattedValue;
                                return `${label}: ${value}%`;
                            }
                        }
                    }
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
        
        // If we need more colors than in our predefined list
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
        const purpose = purposeSelect.value;
        const minArea = parseFloat(minAreaInput.value);
        
        if (purpose === 'Select purpose...' || isNaN(minArea)) {
            suitableLocationsResult.innerHTML = '<p class="text-danger">Please select a valid purpose and minimum area.</p>';
            return;
        }
        
        suitableLocationsResult.innerHTML = '<p class="text-center">Identifying suitable locations...</p>';
        
        fetch('/identify-suitable-locations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                purpose: purpose,
                min_area_sqm: minArea
            })
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    suitableLocationsResult.innerHTML = `<p class="text-danger">${data.message}</p>`;
                    return;
                }
                
                // Display suitable locations
                if (data.suitable_locations && data.suitable_locations.length > 0) {
                    const locationsHTML = data.suitable_locations.map((location, index) => {
                        return `<div class="card mb-2">
                            <div class="card-body">
                                <h6>Location ${index + 1}</h6>
                                <p>Area: ${location.area_sqm.toFixed(2)} sq.m</p>
                                <p>Coordinates: ${location.center.lat.toFixed(6)}, ${location.center.lon.toFixed(6)}</p>
                            </div>
                        </div>`;
                    }).join('');
                    
                    suitableLocationsResult.innerHTML = `
                        <h6>Found ${data.suitable_locations.length} suitable location(s):</h6>
                        ${locationsHTML}
                    `;
                } else {
                    suitableLocationsResult.innerHTML = '<p>No suitable locations found. Try adjusting your criteria.</p>';
                }
            })
            .catch(error => {
                console.error('Error identifying suitable locations:', error);
                suitableLocationsResult.innerHTML = `<p class="text-danger">Error identifying suitable locations. Please try again.</p>`;
            });
    }
    
    // Initialize image mode change listeners
    singleModeRadio.addEventListener('change', function() {
        if (selectedLocation) {
            getSatelliteImage(selectedLocation.lat, selectedLocation.lng);
        }
    });
    
    gridModeRadio.addEventListener('change', function() {
        if (selectedLocation) {
            getSatelliteImage(selectedLocation.lat, selectedLocation.lng);
        }
    });
    
    // Initialize
    locationSearchInput.focus();
});