// Main Application Logic
const API_BASE = window.location.origin;
let currentHospitals = [];
let userPosition = null;
let sessionId = null;

// Initialize app on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Hide loading screen
    setTimeout(() => {
        document.getElementById('loadingScreen').classList.add('hidden');
    }, 1000);
    
    // Get user location
    getUserLocation();
    
    // Setup event listeners
    setupEventListeners();
    
    // Generate session ID
    sessionId = generateSessionId();
}

function setupEventListeners() {
    // Language buttons
    document.querySelectorAll('.language-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setLanguage(btn.dataset.lang);
        });
    });
    
    // Voice button
    document.getElementById('voiceBtn').addEventListener('click', () => {
        if (voiceHandler.isListening) {
            voiceHandler.stopListening();
        } else {
            voiceHandler.startListening();
        }
    });
    
    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', () => {
        findHospitals();
    });
    
    // Emergency button
    document.getElementById('emergencyBtn').addEventListener('click', () => {
        findEmergencyHospitals();
    });
    
    // Enter key in textarea
    document.getElementById('symptomsInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            findHospitals();
        }
    });
    
    // Map controls
    document.getElementById('centerMapBtn')?.addEventListener('click', () => {
        mapHandler.centerOnUser();
    });
    
    // Sort select
    document.getElementById('sortSelect')?.addEventListener('change', (e) => {
        sortHospitals(e.target.value);
    });
    
    // Modal close
    document.getElementById('modalClose')?.addEventListener('click', () => {
        closeModal();
    });
}

function getUserLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                userPosition = {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude
                };
                console.log('User location obtained:', userPosition);
            },
            (error) => {
                console.error('Error getting location:', error);
                // Default to Visakhapatnam
                userPosition = {
                    latitude: 17.6868,
                    longitude: 83.2185
                };
            }
        );
    } else {
        // Default location
        userPosition = {
            latitude: 17.6868,
            longitude: 83.2185
        };
    }
}

async function findHospitals() {
    const symptoms = document.getElementById('symptomsInput').value.trim();
    
    if (!symptoms) {
        alert('Please describe your symptoms');
        return;
    }
    
    if (!userPosition) {
        alert('Unable to get your location. Please enable location services.');
        return;
    }
    
    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Processing...</span>';
    
    try {
        const response = await fetch(`${API_BASE}/api/find-hospitals`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                latitude: userPosition.latitude,
                longitude: userPosition.longitude,
                symptoms: symptoms,
                language: currentLanguage,
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentHospitals = data.hospitals;
            displayAnalysis(data.analysis);
            displayHospitals(data.hospitals);
            
            // Initialize and update map
            mapHandler.initMap(userPosition.latitude, userPosition.longitude);
            mapHandler.addHospitalMarkers(data.hospitals);
        } else {
            alert('Error finding hospitals: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to server');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> <span data-translate="find_hospitals">Find Hospitals</span>';
        updatePageText();
    }
}

async function findEmergencyHospitals() {
    if (!userPosition) {
        alert('Unable to get your location');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/emergency-hospitals`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                latitude: userPosition.latitude,
                longitude: userPosition.longitude,
                language: currentLanguage
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentHospitals = data.hospitals;
            displayHospitals(data.hospitals, true);
            
            mapHandler.initMap(userPosition.latitude, userPosition.longitude);
            mapHandler.addHospitalMarkers(data.hospitals);
            
            // Hide analysis section for emergency
            document.getElementById('analysisSection').style.display = 'none';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to find emergency hospitals');
    }
}

function displayAnalysis(analysis) {
    if (!analysis) return;
    
    const analysisSection = document.getElementById('analysisSection');
    const analysisContent = document.getElementById('analysisContent');
    
    analysisContent.innerHTML = `
        <div class="analysis-item">
            <span class="analysis-label">${translate('category')}:</span>
            <span class="analysis-value">${analysis.category}</span>
        </div>
        <div class="analysis-item">
            <span class="analysis-label">${translate('specialty')}:</span>
            <span class="analysis-value">${analysis.specialty}</span>
        </div>
        <div class="analysis-item">
            <span class="analysis-label">${translate('priority')}:</span>
            <span class="analysis-value" style="color: ${getPriorityColor(analysis.priority)}">${analysis.priority.toUpperCase()}</span>
        </div>
        <div class="analysis-item">
            <span class="analysis-label">${translate('confidence')}:</span>
            <div style="flex: 1;">
                <span class="analysis-value">${(analysis.confidence * 100).toFixed(1)}%</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${analysis.confidence * 100}%"></div>
                </div>
            </div>
        </div>
    `;
    
    analysisSection.style.display = 'block';
    analysisSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayHospitals(hospitals, isEmergency = false) {
    const resultsSection = document.getElementById('resultsSection');
    const hospitalsList = document.getElementById('hospitalsList');
    const noResults = document.getElementById('noResults');
    
    if (hospitals.length === 0) {
        resultsSection.style.display = 'none';
        noResults.style.display = 'block';
        return;
    }
    
    noResults.style.display = 'none';
    resultsSection.style.display = 'block';
    
    hospitalsList.innerHTML = hospitals.map((hospital, index) => `
        <div class="hospital-card" onclick="showHospitalDetails(${hospital.id})">
            <div class="hospital-card-header">
                <div class="hospital-info">
                    <h4>${index + 1}. ${hospital.name}</h4>
                    ${hospital.rating ? `
                        <div class="hospital-rating">
                            <i class="fas fa-star"></i>
                            <span>${hospital.rating.toFixed(1)} (${hospital.total_reviews} reviews)</span>
                        </div>
                    ` : ''}
                </div>
                ${hospital.has_emergency ? `
                    <span class="hospital-badge">
                        <i class="fas fa-ambulance"></i> 24/7 Emergency
                    </span>
                ` : ''}
            </div>
            
            <div class="hospital-details">
                <div class="detail-item">
                    <i class="fas fa-route"></i>
                    <span>${hospital.distance_km} km away</span>
                </div>
                <div class="detail-item">
                    <i class="fas fa-clock"></i>
                    <span>${hospital.travel_time_minutes} min by car</span>
                </div>
                ${!isEmergency && hospital.total_score ? `
                    <div class="detail-item">
                        <i class="fas fa-star"></i>
                        <span>Match Score: ${hospital.total_score}/100</span>
                    </div>
                ` : ''}
            </div>
            
            ${hospital.specialties && hospital.specialties.length > 0 ? `
                <div class="hospital-specialties">
                    ${hospital.specialties.slice(0, 5).map(spec => `
                        <span class="specialty-tag">${spec}</span>
                    `).join('')}
                    ${hospital.specialties.length > 5 ? `
                        <span class="specialty-tag">+${hospital.specialties.length - 5} more</span>
                    ` : ''}
                </div>
            ` : ''}
            
            <div class="hospital-actions">
                <button class="action-btn btn-primary" onclick="event.stopPropagation(); getDirections(${hospital.latitude}, ${hospital.longitude})">
                    <i class="fas fa-directions"></i> ${translate('get_directions')}
                </button>
                ${hospital.phone ? `
                    <button class="action-btn btn-secondary" onclick="event.stopPropagation(); window.location.href='tel:${hospital.phone}'">
                        <i class="fas fa-phone"></i> ${translate('call_now')}
                    </button>
                ` : ''}
            </div>
        </div>
    `).join('');
    
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function showHospitalDetails(hospitalId) {
    try {
        const response = await fetch(`${API_BASE}/api/hospital/${hospitalId}?language=${currentLanguage}`);
        const data = await response.json();
        
        if (data.success) {
            const hospital = data.hospital;
            const modalBody = document.getElementById('modalBody');
            
            modalBody.innerHTML = `
                <h2 style="margin-bottom: 1rem;">${hospital.name}</h2>
                <div style="display: grid; gap: 1.5rem;">
                    <div>
                        <h3 style="color: #2563eb; margin-bottom: 0.5rem;">${translate('contact_info')}</h3>
                        <p><i class="fas fa-map-marker-alt"></i> ${hospital.address}</p>
                        ${hospital.phone ? `<p><i class="fas fa-phone"></i> ${hospital.phone}</p>` : ''}
                        ${hospital.email ? `<p><i class="fas fa-envelope"></i> ${hospital.email}</p>` : ''}
                    </div>
                    
                    ${hospital.specialties && hospital.specialties.length > 0 ? `
                        <div>
                            <h3 style="color: #2563eb; margin-bottom: 0.5rem;">${translate('specialties')}</h3>
                            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                                ${hospital.specialties.map(s => `<span class="specialty-tag">${s}</span>`).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${hospital.facilities && hospital.facilities.length > 0 ? `
                        <div>
                            <h3 style="color: #2563eb; margin-bottom: 0.5rem;">${translate('facilities')}</h3>
                            <ul style="list-style: none; padding: 0;">
                                ${hospital.facilities.map(f => `<li><i class="fas fa-check-circle" style="color: #10b981;"></i> ${f}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
            
            document.getElementById('hospitalModal').classList.add('active');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to load hospital details');
    }
}

function closeModal() {
    document.getElementById('hospitalModal').classList.remove('active');
}

function getDirections(lat, lng) {
    // Open in Google Maps (free, no API key needed for this)
    const url = `https://www.google.com/maps/dir/?api=1&origin=${userPosition.latitude},${userPosition.longitude}&destination=${lat},${lng}`;
    window.open(url, '_blank');
    
    // Also show route on our map
    mapHandler.showRoute(lat, lng);
}

function sortHospitals(sortBy) {
    if (!currentHospitals || currentHospitals.length === 0) return;
    
    let sorted = [...currentHospitals];
    
    switch(sortBy) {
        case 'distance':
            sorted.sort((a, b) => a.distance_km - b.distance_km);
            break;
        case 'rating':
            sorted.sort((a, b) => (b.rating || 0) - (a.rating || 0));
            break;
        case 'score':
        default:
            sorted.sort((a, b) => (b.total_score || 0) - (a.total_score || 0));
    }
    
    currentHospitals = sorted;
    displayHospitals(sorted);
    mapHandler.addHospitalMarkers(sorted);
}

function getPriorityColor(priority) {
    const colors = {
        'critical': '#ef4444',
        'urgent': '#f59e0b',
        'normal': '#10b981'
    };
    return colors[priority] || '#10b981';
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('hospitalModal');
    if (event.target === modal) {
        closeModal();
    }
}