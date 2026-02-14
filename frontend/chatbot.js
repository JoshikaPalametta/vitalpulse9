// Chatbot functionality
class MedicalChatbot {
    constructor() {
        this.chatContainer = document.getElementById('chatbot-container');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.toggleBtn = document.getElementById('chat-toggle');
        this.closeBtn = document.getElementById('close-chat');
        
        this.init();
    }
    
    init() {
        this.toggleBtn.addEventListener('click', () => this.toggleChat());
        this.closeBtn.addEventListener('click', () => this.toggleChat());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        // Welcome message
        this.addBotMessage('Hello! I\'m your medical assistant. How can I help you today?');
    }
    
    toggleChat() {
        this.chatContainer.classList.toggle('chatbot-hidden');
    }
    
    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Add user message
        this.addUserMessage(message);
        this.chatInput.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch('http://localhost:5000/api/chatbot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator();
            
            // Add bot response
            this.addBotMessage(data.response);
            
        } catch (error) {
            this.removeTypingIndicator();
            this.addBotMessage('Sorry, I encountered an error. Please try again.');
            console.error('Chat error:', error);
        }
    }
    
    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.textContent = text;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addBotMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.textContent = text;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message bot typing-indicator';
        indicator.id = 'typing';
        indicator.textContent = '...';
        this.chatMessages.appendChild(indicator);
        this.scrollToBottom();
    }
    
    removeTypingIndicator() {
        const indicator = document.getElementById('typing');
        if (indicator) indicator.remove();
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new MedicalChatbot();
});