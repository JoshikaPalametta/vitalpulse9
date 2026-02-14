"""
Main Flask Application for Voice-Guided AI Hospital Finder
WITH ADVANCED AI SYMPTOM ANALYZER (90%+ Accuracy)
"""


import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import tempfile
from models import db, Hospital, SearchHistory, Specialty
from advanced_symptom_analyzer import advanced_symptom_analyzer as symptom_analyzer
from hospital_recommender import hospital_recommender

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///hospital_finder.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()


# ============= ROUTES =============

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',  # Updated version with advanced AI
        'ai_model': 'advanced' if 'advanced' in str(type(symptom_analyzer)) else 'basic'
    })


@app.route('/api/analyze-symptoms', methods=['POST'])
@app.route('/api/analyze', methods=['POST'])   # alias route — same function, no duplicate
def analyze_symptoms():
    """
    Analyze symptoms and return classification using ADVANCED AI
    
    Request body:
    {
        "symptoms": "chest pain and shortness of breath",
        "language": "en"  # optional - auto-detected if not provided
    }
    """
    try:
        data = request.get_json()
        symptoms_text = data.get('symptoms', '').strip()
        language = data.get('language', None)
        
        if not symptoms_text:
            return jsonify({'error': 'Symptoms text is required'}), 400
        
        # Analyze symptoms with advanced AI
        analysis_result = symptom_analyzer.analyze_symptoms(symptoms_text, language)
        
        # Get related specialties
        related_specialties = symptom_analyzer.get_related_specialties(
            analysis_result['category']
        )
        analysis_result['related_specialties'] = related_specialties
        
        # Add model info
        analysis_result['model_type'] = 'advanced'
        analysis_result['model_version'] = '2.0'
        
        return jsonify({
            'success': True,
            'analysis': analysis_result
        })
    
    except Exception as e:
        app.logger.error(f"Error analyzing symptoms: {str(e)}")
        return jsonify({'error': 'Failed to analyze symptoms'}), 500


@app.route('/api/find-hospitals', methods=['POST'])
def find_hospitals():
    """
    Find nearby hospitals based on location and symptoms
    Uses ADVANCED AI for symptom analysis
    
    Request body:
    {
        "latitude": 17.6868,
        "longitude": 83.2185,
        "symptoms": "chest pain",
        "language": "en",
        "max_distance": 50  # optional
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        symptoms = data.get('symptoms', '').strip()
        language = data.get('language', 'en')
        max_distance = data.get('max_distance', 50)
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'Location coordinates are required'}), 400
        
        # Analyze symptoms if provided (using ADVANCED AI)
        analysis_result = None
        required_specialties = []
        priority = 'normal'
        
        if symptoms:
            analysis_result = symptom_analyzer.analyze_symptoms(symptoms, language)
            required_specialties = symptom_analyzer.get_related_specialties(
                analysis_result['category']
            )
            priority = analysis_result['priority']
            
            # Add model metadata
            analysis_result['model_type'] = 'advanced'
            analysis_result['model_version'] = '2.0'
        
        # Find hospitals (with improved distance-based ranking)
        hospitals = hospital_recommender.find_nearby_hospitals(
            user_lat=latitude,
            user_lon=longitude,
            required_specialties=required_specialties,
            priority=priority,
            limit=10,
            language=language
        )
        
        # Save search history
        session_id = data.get('session_id', str(uuid.uuid4()))
        if symptoms and hospitals:
            search_record = SearchHistory(
                session_id=session_id,
                symptoms=symptoms,
                language=language,
                user_latitude=latitude,
                user_longitude=longitude,
                recommended_hospital_id=hospitals[0]['id'] if hospitals else None,
                predicted_category=analysis_result['category'] if analysis_result else None,
                confidence_score=analysis_result['confidence'] if analysis_result else None
            )
            db.session.add(search_record)
            db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'hospitals': hospitals,
            'total_found': len(hospitals),
            'session_id': session_id
        })
    
    except Exception as e:
        app.logger.error(f"Error finding hospitals: {str(e)}")
        return jsonify({'error': 'Failed to find hospitals'}), 500


@app.route('/api/emergency-hospitals', methods=['POST'])
def emergency_hospitals():
    """
    Get nearest emergency hospitals for critical situations
    
    Request body:
    {
        "latitude": 17.6868,
        "longitude": 83.2185,
        "language": "en"
    }
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        language = data.get('language', 'en')
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'Location coordinates are required'}), 400
        
        hospitals = hospital_recommender.get_emergency_hospitals(
            user_lat=latitude,
            user_lon=longitude,
            language=language
        )
        
        return jsonify({
            'success': True,
            'hospitals': hospitals,
            'total_found': len(hospitals)
        })
    
    except Exception as e:
        app.logger.error(f"Error finding emergency hospitals: {str(e)}")
        return jsonify({'error': 'Failed to find emergency hospitals'}), 500


@app.route('/api/hospital/<int:hospital_id>', methods=['GET'])
def get_hospital_details(hospital_id):
    """Get detailed information about a specific hospital"""
    try:
        language = request.args.get('language', 'en')
        hospital = Hospital.query.get(hospital_id)
        
        if not hospital:
            return jsonify({'error': 'Hospital not found'}), 404
        
        return jsonify({
            'success': True,
            'hospital': hospital.to_dict(language)
        })
    
    except Exception as e:
        app.logger.error(f"Error getting hospital details: {str(e)}")
        return jsonify({'error': 'Failed to get hospital details'}), 500


@app.route('/api/voice-to-text', methods=['POST'])
def voice_to_text():
    """
    Convert voice audio to text
    
    Expects audio file in request
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en-US')
        
        # Language mapping for speech recognition
        language_map = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'te': 'te-IN'
        }
        
        if language in language_map:
            language = language_map[language]
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            
            # Recognize speech
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio.name) as source:
                audio_data = recognizer.record(source)
                
                try:
                    text = recognizer.recognize_google(audio_data, language=language)
                    os.unlink(temp_audio.name)
                    
                    return jsonify({
                        'success': True,
                        'text': text,
                        'language': language
                    })
                
                except sr.UnknownValueError:
                    os.unlink(temp_audio.name)
                    return jsonify({'error': 'Could not understand audio'}), 400
                
                except sr.RequestError as e:
                    os.unlink(temp_audio.name)
                    return jsonify({'error': f'Speech recognition error: {str(e)}'}), 500
    
    except Exception as e:
        app.logger.error(f"Error in voice-to-text: {str(e)}")
        return jsonify({'error': 'Failed to process audio'}), 500


@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech
    
    Request body:
    {
        "text": "Nearest hospital is 2 km away",
        "language": "en"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate speech
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        
        return send_from_directory(
            os.path.dirname(temp_file.name),
            os.path.basename(temp_file.name),
            mimetype='audio/mpeg'
        )
    
    except Exception as e:
        app.logger.error(f"Error in text-to-speech: {str(e)}")
        return jsonify({'error': 'Failed to generate speech'}), 500


@app.route('/api/search-history/<session_id>', methods=['GET'])
def get_search_history(session_id):
    """Get search history for a session"""
    try:
        history = SearchHistory.query.filter_by(session_id=session_id).order_by(
            SearchHistory.searched_at.desc()
        ).all()
        
        history_data = []
        for record in history:
            history_data.append({
                'id': record.id,
                'symptoms': record.symptoms,
                'language': record.language,
                'predicted_category': record.predicted_category,
                'confidence_score': record.confidence_score,
                'searched_at': record.searched_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'history': history_data
        })
    
    except Exception as e:
        app.logger.error(f"Error getting search history: {str(e)}")
        return jsonify({'error': 'Failed to get search history'}), 500


@app.route('/api/specialties', methods=['GET'])
def get_specialties():
    """Get list of all medical specialties"""
    try:
        language = request.args.get('language', 'en')
        specialties = Specialty.query.all()
        
        specialty_list = []
        for specialty in specialties:
            name_field = 'name'
            if language == 'te' and specialty.name_te:
                name_field = 'name_te'
            elif language == 'hi' and specialty.name_hi:
                name_field = 'name_hi'
            
            specialty_list.append({
                'id': specialty.id,
                'name': getattr(specialty, name_field),
                'description': specialty.description
            })
        
        return jsonify({
            'success': True,
            'specialties': specialty_list
        })
    
    except Exception as e:
        app.logger.error(f"Error getting specialties: {str(e)}")
        return jsonify({'error': 'Failed to get specialties'}), 500


# NEW ENDPOINT: Train advanced model (admin only - secure this in production)
@app.route('/api/admin/train-model', methods=['POST'])
def train_model():
    """
    Train the advanced AI model
    
    WARNING: This takes 5-10 minutes. Only call when needed.
    In production, secure this endpoint with authentication.
    """
    try:
        # Check for admin key (basic security)
        admin_key = request.headers.get('X-Admin-Key')
        if admin_key != os.getenv('ADMIN_KEY', 'dev-admin-key'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Train the model
        print("🚀 Starting advanced model training...")
        symptom_analyzer._train_advanced_model()
        
        return jsonify({
            'success': True,
            'message': 'Advanced model trained successfully',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        app.logger.error(f"Error training model: {str(e)}")
        return jsonify({'error': 'Failed to train model'}), 500

# Add this CHATBOT endpoint to your app.py (around line 480, BEFORE the existing /api/analyze route)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """Chatbot endpoint for conversational interface"""
    try:
        data = request.json
        user_message = data.get('message', '').lower()
        
        # Simple intent recognition
        if any(word in user_message for word in ['hello', 'hi', 'hey', 'start']):
            return jsonify({
                'response': 'Hello! I\'m your medical assistant. I can help analyze symptoms and recommend hospitals. What symptoms are you experiencing?',
                'type': 'greeting'
            })
        
        elif any(word in user_message for word in ['symptom', 'feeling', 'pain', 'sick', 'ill', 'disease']):
            return jsonify({
                'response': 'Please describe your symptoms in detail. You can use the symptom checker form above, or tell me what you\'re experiencing.',
                'type': 'symptom_query'
            })
        
        elif any(word in user_message for word in ['hospital', 'doctor', 'clinic', 'recommend']):
            return jsonify({
                'response': 'I can recommend hospitals based on your condition. First, let me know your symptoms so I can provide the best recommendations.',
                'type': 'hospital_query'
            })
        
        
        elif any(word in user_message for word in ['thank', 'thanks', 'bye', 'goodbye']):
            return jsonify({
                'response': 'You\'re welcome! Take care of your health. Remember to consult a doctor if symptoms persist or worsen.',
                'type': 'farewell'
            })
        
        elif any(word in user_message for word in ['help', 'what can you do', 'how']):
            return jsonify({
                'response': 'I can help you: \n• Analyze your symptoms\n• Predict potential diseases\n• Recommend appropriate hospitals\n• Answer health-related questions\n\nJust tell me what you need!',
                'type': 'help'
            })
        
        else:
            # Default response for unclear queries
            return jsonify({
                'response': 'I understand you need help. Please use the symptom checker form above for accurate diagnosis, or ask me about symptoms, hospitals, or how I can assist you.',
                'type': 'redirect'
            })
            
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Your existing /api/analyze route stays as it is - DON'T TOUCH IT!
# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print("\n" + "="*70)
    print("  🏥 AI HOSPITAL FINDER - Starting Server")
    print("="*70)
    print(f"  🌐 Server: http://{host}:{port}")
    print(f"  🤖 AI Model: {'Advanced (90%+ accuracy)' if 'advanced' in str(type(symptom_analyzer)) else 'Basic'}")
    print(f"  🗣️  Languages: English, Hindi, Telugu")
    print(f"  🚨 Emergency Mode: Enabled")
    print("="*70 + "\n")
    
    app.run(host=host, port=port, debug=debug)