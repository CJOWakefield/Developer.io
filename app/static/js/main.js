let map, drawnItems, selectedLocation;
document.addEventListener('DOMContentLoaded', function() {
    // Initialize map
    map = L.map('map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    // Initialize drawing controls
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    const drawControl = new L.Control.Draw({
        draw: {
            polygon: true,
            rectangle: true,
            circle: false,
            circlemarker: false,
            marker: false,
            polyline: false
        },
        edit: {
            featureGroup: drawnItems
        }
    });
    map.addControl(drawControl);
    // Location search
    const searchInput = document.getElementById('locationSearch');
    const suggestionsDiv = document.getElementById('suggestions');
    
    searchInput.addEventListener('input', debounce(async function() {
        if (this.value.length < 3) return;
        
        try {
            const response = await fetch(`/search-location?query=${encodeURIComponent(this.value)}`);
            const suggestions = await response.json();
            
            suggestionsDiv.innerHTML = '';
            suggestionsDiv.style.display = 'block';
            
            suggestions.forEach(suggestion => {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                div.textContent = suggestion.name;
                div.addEventListener('click', () => selectLocation(suggestion));
                suggestionsDiv.appendChild(div);
            });
        } catch (error) {
            console.error('Error fetching suggestions:', error);
        }
    }, 300));
    // Drawing events
    map.on('draw:created', function(e) {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        document.getElementById('analyzeButton').disabled = false;
    });
    // Analyze button
    document.getElementById('analyzeButton').addEventListener('click', async function() {
        if (!selectedLocation || drawnItems.getLayers().length === 0) return;
        
        const bounds = drawnItems.getBounds();
        
        try {
            const response = await fetch('/get-region-data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    bounds: {
                        north: bounds.getNorth(),
                        south: bounds.getSouth(),
                        east: bounds.getEast(),
                        west: bounds.getWest()
                    },
                    location: selectedLocation
                })
            });
            
            const data = await response.json();
            displayResults(data.predictions);
        } catch (error) {
            console.error('Error analyzing region:', error);
        }
    });
});
function selectLocation(location) {
    selectedLocation = location;
    document.getElementById('locationSearch').value = location.name;
    document.getElementById('suggestions').style.display = 'none';
    document.getElementById('analysisControls').classList.remove('hidden');
    
    map.setView([location.lat, location.lng], 12);
}
async function displayResults(predictions) {
    const resultsDiv = document.getElementById('results');
    const predictionResultsDiv = document.getElementById('predictionResults');
    
    resultsDiv.classList.remove('hidden');
    predictionResultsDiv.innerHTML = '';
    
    // Create results display
    const resultHTML = `
        <div class="prediction-container">
            <div class="image-grid">
                ${predictions.image_paths.map((path, index) => `
                    <div class="image-pair">
                        <div class="original-image">
                            <h3>Original Image</h3>
                            <img src="${path}" alt="Original ${index + 1}">
                        </div>
                        <div class="prediction-image">
                            <h3>Prediction</h3>
                            <img src="${predictions.prediction_paths[index]}" alt="Prediction ${index + 1}">
                        </div>
                    </div>
                `).join('')}
            </div>
            ${predictions.metrics ? `
                <div class="metrics">
                    <h3>Analysis Metrics</h3>
                    <ul>
                        ${Object.entries(predictions.metrics).map(([key, value]) => `
                            <li><strong>${key}:</strong> ${value}</li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
        </div>
    `;
    
    predictionResultsDiv.innerHTML = resultHTML;
}
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
} 