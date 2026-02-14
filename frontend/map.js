// Map Handler using Leaflet.js with FREE OpenStreetMap
// No API key required! Completely free!

class MapHandler {
    constructor() {
        this.map = null;
        this.userMarker = null;
        this.hospitalMarkers = [];
        this.userLocation = null;
        this.routingControl = null;
    }
    
    initMap(lat, lng) {
        if (!this.map) {
            // Create map with OpenStreetMap tiles (FREE!)
            this.map = L.map('map').setView([lat, lng], 13);
            
            // Add free OpenStreetMap tiles
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom: 19,
                minZoom: 3
            }).addTo(this.map);
            
            // Add scale control
            L.control.scale().addTo(this.map);
        }
        
        this.setUserLocation(lat, lng);
    }
    
    setUserLocation(lat, lng) {
        this.userLocation = {lat, lng};
        
        if (this.userMarker) {
            this.userMarker.setLatLng([lat, lng]);
        } else {
            // Custom user location marker
            const userIcon = L.divIcon({
                className: 'user-marker',
                html: `<div style="background: #2563eb; width: 24px; height: 24px; border-radius: 50%; border: 4px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.4); animation: pulse 2s infinite;"></div>`,
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            });
            
            this.userMarker = L.marker([lat, lng], {icon: userIcon})
                .addTo(this.map)
                .bindPopup('<b>📍 Your Location</b>')
                .openPopup();
        }
        
        this.map.setView([lat, lng], 13);
    }
    
    addHospitalMarkers(hospitals) {
        // Clear existing markers and routes
        this.hospitalMarkers.forEach(marker => marker.remove());
        this.hospitalMarkers = [];
        if (this.routingControl) {
            this.map.removeControl(this.routingControl);
            this.routingControl = null;
        }
        
        hospitals.forEach((hospital, index) => {
            // Custom hospital marker with ranking
            const color = index === 0 ? '#ef4444' : (index < 3 ? '#f59e0b' : '#10b981');
            const hospitalIcon = L.divIcon({
                className: 'hospital-marker',
                html: `
                    <div style="
                        background: ${color}; 
                        color: white; 
                        padding: 8px 12px; 
                        border-radius: 12px; 
                        font-weight: bold; 
                        font-size: 14px;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.3);
                        border: 2px solid white;
                    ">
                        ${index + 1}
                    </div>
                `,
                iconSize: [40, 40],
                iconAnchor: [20, 20]
            });
            
            const marker = L.marker([hospital.latitude, hospital.longitude], {
                icon: hospitalIcon
            }).addTo(this.map);
            
            // Detailed popup
            const popupContent = `
                <div style="min-width: 250px; font-family: Arial, sans-serif;">
                    <h3 style="margin: 0 0 10px 0; color: #2563eb; font-size: 16px;">
                        ${index + 1}. ${hospital.name}
                    </h3>
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #059669;">📍 ${hospital.distance_km} km away</strong><br>
                        <span style="font-size: 13px; color: #666;">🚗 ${hospital.travel_time_minutes} min by car</span>
                    </div>
                    ${hospital.rating ? `
                        <div style="margin-bottom: 8px; font-size: 13px;">
                            ⭐ ${hospital.rating.toFixed(1)}/5.0
                        </div>
                    ` : ''}
                    ${hospital.has_emergency ? `
                        <div style="background: #ef4444; color: white; padding: 4px 8px; border-radius: 4px; display: inline-block; font-size: 11px; margin-bottom: 8px;">
                            🚨 24/7 Emergency
                        </div>
                    ` : ''}
                    <div style="display: grid; gap: 6px; margin-top: 10px;">
                        <button 
                            onclick="showHospitalDetails(${hospital.id})" 
                            style="width: 100%; padding: 8px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">
                            📋 View Details
                        </button>
                        <button 
                            onclick="mapHandler.showRoute(${hospital.latitude}, ${hospital.longitude})" 
                            style="width: 100%; padding: 8px; background: #059669; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">
                            🗺️ Show Route
                        </button>
                    </div>
                </div>
            `;
            
            marker.bindPopup(popupContent, {maxWidth: 300});
            this.hospitalMarkers.push(marker);
            
            // Auto-open popup for top result
            if (index === 0) {
                marker.openPopup();
            }
        });
        
        // Fit map to show all markers
        if (hospitals.length > 0) {
            const bounds = L.latLngBounds(
                hospitals.map(h => [h.latitude, h.longitude])
                    .concat([[this.userLocation.lat, this.userLocation.lng]])
            );
            this.map.fitBounds(bounds, {padding: [50, 50]});
        }
    }
    
    showRoute(hospitalLat, hospitalLng) {
        // Remove existing route if any
        if (this.routingControl) {
            this.map.removeControl(this.routingControl);
        }
        
        // Draw a simple line from user to hospital
        const routeLine = L.polyline([
            [this.userLocation.lat, this.userLocation.lng],
            [hospitalLat, hospitalLng]
        ], {
            color: '#2563eb',
            weight: 4,
            opacity: 0.7,
            dashArray: '10, 10'
        }).addTo(this.map);
        
        this.routingControl = routeLine;
        
        // Fit bounds to show the route
        this.map.fitBounds(routeLine.getBounds(), {padding: [50, 50]});
    }
    
    centerOnUser() {
        if (this.userLocation) {
            this.map.setView([this.userLocation.lat, this.userLocation.lng], 14, {
                animate: true,
                duration: 1
            });
        }
    }
}

const mapHandler = new MapHandler();