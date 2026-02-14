"""
Database Models for Hospital Finder Application
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Float, Integer, String, Text, Boolean, DateTime, JSON

db = SQLAlchemy()

class Hospital(db.Model):
    """Hospital model with detailed information"""
    __tablename__ = 'hospitals'
    
    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(200), nullable=False)
    name_te = db.Column(String(200))  # Telugu name
    name_hi = db.Column(String(200))  # Hindi name
    
    # Location
    latitude = db.Column(Float, nullable=False)
    longitude = db.Column(Float, nullable=False)
    address = db.Column(Text, nullable=False)
    address_te = db.Column(Text)
    address_hi = db.Column(Text)
    city = db.Column(String(100))
    state = db.Column(String(100))
    pincode = db.Column(String(10))
    
    # Contact
    phone = db.Column(String(20))
    emergency_phone = db.Column(String(20))
    email = db.Column(String(100))
    website = db.Column(String(200))
    
    # Specialties and Services
    specialties = db.Column(JSON)  # List of specialties
    services = db.Column(JSON)     # Available services
    facilities = db.Column(JSON)   # Facilities like ICU, Emergency, etc.
    
    # Operational Details
    is_24x7 = db.Column(Boolean, default=True)
    has_emergency = db.Column(Boolean, default=True)
    has_ambulance = db.Column(Boolean, default=False)
    bed_capacity = db.Column(Integer)
    
    # Ratings
    rating = db.Column(Float, default=0.0)
    total_reviews = db.Column(Integer, default=0)
    
    # Status
    is_active = db.Column(Boolean, default=True)
    verified = db.Column(Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Hospital {self.name}>'
    
    def to_dict(self, language='en'):
        """Convert hospital to dictionary with language support"""
        name_field = 'name'
        address_field = 'address'
        
        if language == 'te':
            name_field = 'name_te' if self.name_te else 'name'
            address_field = 'address_te' if self.address_te else 'address'
        elif language == 'hi':
            name_field = 'name_hi' if self.name_hi else 'name'
            address_field = 'address_hi' if self.address_hi else 'address'
        
        return {
            'id': self.id,
            'name': getattr(self, name_field),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'address': getattr(self, address_field),
            'city': self.city,
            'state': self.state,
            'pincode': self.pincode,
            'phone': self.phone,
            'emergency_phone': self.emergency_phone,
            'email': self.email,
            'website': self.website,
            'specialties': self.specialties or [],
            'services': self.services or [],
            'facilities': self.facilities or [],
            'is_24x7': self.is_24x7,
            'has_emergency': self.has_emergency,
            'has_ambulance': self.has_ambulance,
            'bed_capacity': self.bed_capacity,
            'rating': self.rating,
            'total_reviews': self.total_reviews,
        }


class SearchHistory(db.Model):
    """Track user search history for personalization"""
    __tablename__ = 'search_history'
    
    id = db.Column(Integer, primary_key=True)
    session_id = db.Column(String(100))
    
    # Search Details
    symptoms = db.Column(Text)
    language = db.Column(String(10))
    user_latitude = db.Column(Float)
    user_longitude = db.Column(Float)
    
    # Results
    recommended_hospital_id = db.Column(Integer, db.ForeignKey('hospitals.id'))
    selected_hospital_id = db.Column(Integer, db.ForeignKey('hospitals.id'))
    
    # Classification
    predicted_category = db.Column(String(100))
    confidence_score = db.Column(Float)
    
    # Timestamp
    searched_at = db.Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SearchHistory {self.id}>'


class Specialty(db.Model):
    """Medical specialties master data"""
    __tablename__ = 'specialties'
    
    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(100), nullable=False, unique=True)
    name_te = db.Column(String(100))
    name_hi = db.Column(String(100))
    description = db.Column(Text)
    keywords = db.Column(JSON)  # Keywords for matching
    
    def __repr__(self):
        return f'<Specialty {self.name}>'


class SymptomCategory(db.Model):
    """Symptom categories for classification"""
    __tablename__ = 'symptom_categories'
    
    id = db.Column(Integer, primary_key=True)
    category = db.Column(String(100), nullable=False, unique=True)
    category_te = db.Column(String(100))
    category_hi = db.Column(String(100))
    description = db.Column(Text)
    keywords = db.Column(JSON)  # Symptom keywords
    related_specialties = db.Column(JSON)  # Related medical specialties
    priority_level = db.Column(String(20))  # emergency, urgent, normal
    
    def __repr__(self):
        return f'<SymptomCategory {self.category}>'