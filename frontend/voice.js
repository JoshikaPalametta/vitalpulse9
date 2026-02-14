// Voice Recognition Handler
class VoiceHandler {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.initRecognition();
    }
    
    initRecognition() {
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = true;
            
            this.recognition.onstart = () => {
                this.isListening = true;
                this.showListening();
            };
            
            this.recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0].transcript)
                    .join('');
                
                document.getElementById('symptomsInput').value = transcript;
                
                if (event.results[0].isFinal) {
                    this.stopListening();
                }
            };
            
            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.stopListening();
                alert('Error: ' + event.error);
            };
            
            this.recognition.onend = () => {
                this.stopListening();
            };
        }
    }
    
    setLanguage(lang) {
        if (this.recognition) {
            const langMap = {
                'en': 'en-US',
                'hi': 'hi-IN',
                'te': 'te-IN'
            };
            this.recognition.lang = langMap[lang] || 'en-US';
        }
    }
    
    startListening() {
        if (this.recognition && !this.isListening) {
            this.setLanguage(currentLanguage);
            this.recognition.start();
        }
    }
    
    stopListening() {
        if (this.recognition && this.isListening) {
            this.isListening = false;
            this.recognition.stop();
            this.hideListening();
        }
    }
    
    showListening() {
        const voiceBtn = document.getElementById('voiceBtn');
        const voiceFeedback = document.getElementById('voiceFeedback');
        const feedbackText = document.getElementById('feedbackText');
        
        voiceBtn.classList.add('listening');
        voiceFeedback.style.display = 'block';
        feedbackText.textContent = translate('listening');
    }
    
    hideListening() {
        const voiceBtn = document.getElementById('voiceBtn');
        const voiceFeedback = document.getElementById('voiceFeedback');
        
        voiceBtn.classList.remove('listening');
        setTimeout(() => {
            voiceFeedback.style.display = 'none';
        }, 500);
    }
}

const voiceHandler = new VoiceHandler();