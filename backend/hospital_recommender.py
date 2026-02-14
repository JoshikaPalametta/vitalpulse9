"""
IMPROVED Hospital Recommendation System
Prioritizes NEAREST hospitals first, then applies specialty/rating filters
"""
import math
from typing import List, Dict
from geopy.distance import geodesic
from models import Hospital, db


class HospitalRecommender:
    
    """
    Recommends hospitals with DISTANCE as the ONLY sort factor:
    1. DISTANCE — Always sorted nearest to farthest (strict)
    2. Total score (specialty + rating + emergency) used ONLY as tiebreaker
       when two hospitals are at the exact same distance
    """
    
    def __init__(self, max_distance_km=50):
        self.max_distance_km = max_distance_km
    
    def calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates in kilometers
        """
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    
    def calculate_travel_time(self, distance_km: float, 
                             transport_mode: str = 'car') -> int:
        """
        Estimate travel time in minutes based on distance and transport mode
        """
        # Average speeds in km/h
        speeds = {
            'car': 40,
            'bike': 25,
            'walk': 5,
            'ambulance': 60
        }
        
        speed = speeds.get(transport_mode, 40)
        time_hours = distance_km / speed
        return int(time_hours * 60)
    
    def score_hospital(self, hospital: Hospital, user_lat: float, user_lon: float,
                      required_specialties: List[str], priority: str = 'normal') -> Dict:
        """
        Calculate score with DISTANCE as PRIMARY factor
        
        Scoring:
        - Critical/Emergency: 80% distance, 10% specialty, 10% emergency availability
        - Urgent: 70% distance, 15% specialty, 10% rating, 5% facilities
        - Normal: 60% distance, 20% specialty, 15% rating, 5% facilities
        """
        # Calculate distance
        distance = self.calculate_distance(
            user_lat, user_lon,
            hospital.latitude, hospital.longitude
        )
        
        # Skip if too far
        if distance > self.max_distance_km:
            return None
        
        # Initialize scores
        scores = {
            'distance_score': 0,
            'specialty_score': 0,
            'rating_score': 0,
            'emergency_score': 0,
            'facility_score': 0
        }
        
        # Determine weights based on priority
        if priority == 'critical':
            weights = {'distance': 80, 'specialty': 0, 'rating': 0, 'emergency': 20, 'facility': 0}
        elif priority == 'urgent':
            weights = {'distance': 70, 'specialty': 15, 'rating': 10, 'emergency': 5, 'facility': 0}
        else:
            weights = {'distance': 60, 'specialty': 20, 'rating': 15, 'emergency': 0, 'facility': 5}
        
        # 1. DISTANCE SCORE (PRIMARY FACTOR)
        # Use exponential decay for better differentiation of nearby hospitals
        # Closer = much higher score
        distance_normalized = min(distance / self.max_distance_km, 1)
        
        # Exponential scoring: very close hospitals get disproportionately higher scores
        # This ensures 1km hospital always beats 5km hospital
        distance_score = weights['distance'] * math.exp(-3 * distance_normalized)
        scores['distance_score'] = distance_score
        
        # 2. Specialty match score
        if weights['specialty'] > 0:
            hospital_specialties = [s.lower() for s in (hospital.specialties or [])]
            required_specialties_lower = [s.lower() for s in required_specialties]
            
            matches = sum(1 for spec in required_specialties_lower 
                         if any(spec in h_spec for h_spec in hospital_specialties))
            
            if required_specialties:
                specialty_match_ratio = matches / len(required_specialties)
                scores['specialty_score'] = weights['specialty'] * specialty_match_ratio
            else:
                scores['specialty_score'] = weights['specialty'] * 0.5  # Neutral score
        
        # 3. Rating score
        if weights['rating'] > 0:
            rating_score = (hospital.rating / 5.0) * weights['rating'] if hospital.rating else weights['rating'] * 0.5
            scores['rating_score'] = rating_score
        
        # 4. Emergency availability score
        if weights['emergency'] > 0:
            emergency_points = 0
            if hospital.has_emergency:
                emergency_points += 0.5
            if hospital.is_24x7:
                emergency_points += 0.3
            if hospital.has_ambulance:
                emergency_points += 0.2
            scores['emergency_score'] = weights['emergency'] * min(emergency_points, 1.0)
        
        # 5. Facility score
        if weights['facility'] > 0:
            facilities = hospital.facilities or []
            facility_keywords = ['icu', 'emergency', 'operation theater', 'trauma center']
            facility_count = sum(1 for keyword in facility_keywords 
                                if any(keyword in str(f).lower() for f in facilities))
            facility_ratio = min(facility_count / len(facility_keywords), 1.0)
            scores['facility_score'] = weights['facility'] * facility_ratio
        
        # Calculate total score
        total_score = sum(scores.values())
        
        # Calculate travel times
        travel_time = self.calculate_travel_time(distance, 'car')
        travel_time_ambulance = self.calculate_travel_time(distance, 'ambulance')
        
        return {
            'hospital_id': hospital.id,
            'distance_km': round(distance, 2),
            'travel_time_minutes': travel_time,
            'travel_time_ambulance_minutes': travel_time_ambulance,
            'total_score': round(total_score, 2),
            'score_breakdown': {k: round(v, 2) for k, v in scores.items()},
            'specialty_match': scores.get('specialty_score', 0) > 0,
            'priority': priority
        }
    
    def find_nearby_hospitals(self, user_lat: float, user_lon: float,
                             required_specialties: List[str] = None,
                             priority: str = 'normal',
                             limit: int = 10,
                             language: str = 'en') -> List[Dict]:
        """
        Find and rank nearby hospitals — STRICTLY NEAREST FIRST
        
        Sort order:
          1. distance_km ascending (closest is always #1)
          2. total_score descending (tiebreaker only)
        """
        # Get all active hospitals
        hospitals = Hospital.query.filter_by(is_active=True).all()
        
        if not hospitals:
            return []
        
        # Score each hospital
        scored_hospitals = []
        for hospital in hospitals:
            score_data = self.score_hospital(
                hospital, user_lat, user_lon,
                required_specialties or [],
                priority
            )
            
            if score_data:
                # Combine hospital data with score
                hospital_dict = hospital.to_dict(language)
                hospital_dict.update(score_data)
                scored_hospitals.append(hospital_dict)
        
        # Sort STRICTLY by distance first — closest hospital always comes first
        # Total score only used as tiebreaker when two hospitals are same distance
        scored_hospitals.sort(key=lambda x: (x['distance_km'], -x['total_score']))
        
        # Return top results
        return scored_hospitals[:limit]
    
    def get_emergency_hospitals(self, user_lat: float, user_lon: float,
                               max_distance: float = 20,
                               language: str = 'en') -> List[Dict]:
        """
        Get nearest emergency hospitals (for critical situations)
        SORTED BY DISTANCE ONLY
        """
        hospitals = Hospital.query.filter_by(
            is_active=True,
            has_emergency=True
        ).all()
        
        emergency_hospitals = []
        for hospital in hospitals:
            distance = self.calculate_distance(
                user_lat, user_lon,
                hospital.latitude, hospital.longitude
            )
            
            if distance <= max_distance:
                hospital_dict = hospital.to_dict(language)
                hospital_dict['distance_km'] = round(distance, 2)
                hospital_dict['travel_time_ambulance'] = self.calculate_travel_time(
                    distance, 'ambulance'
                )
                emergency_hospitals.append(hospital_dict)
        
        # Sort ONLY by distance for emergencies
        emergency_hospitals.sort(key=lambda x: x['distance_km'])
        
        return emergency_hospitals[:5]  # Return top 5 nearest


# Singleton instance
hospital_recommender = HospitalRecommender()