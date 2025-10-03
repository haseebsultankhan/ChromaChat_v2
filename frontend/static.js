// NADRA Agent Chat Interface
// Clean, professional implementation with dark/light mode support

class NADRAChat {
  constructor() {
    this.mediaRecorder = null;
    this.recordingChunks = [];
    this.recordingStartTime = 0;
    this.recordingTimer = null;
    this.maxRecordingTime = 15000; // 15 seconds

    this.initializeElements();
    this.bindEvents();
  }

  initializeElements() {
    // Main elements
    this.chatToggle = document.getElementById('chat-toggle');
    this.chatWrapper = document.getElementById('chat-wrapper');
    this.chatBody = document.getElementById('chat-body');
    this.newChatBtn = document.getElementById('new-chat-btn');
    this.closeChatBtn = document.getElementById('close-chat-btn');

    // Input elements
    this.inputArea = document.getElementById('input-area');
    this.chatInput = document.getElementById('chat-input');
    this.micBtn = document.getElementById('mic-btn');
    this.sendBtn = document.getElementById('send-btn');

    // Recording elements
    this.recordingArea = document.getElementById('recording-area');
    this.recordingTimer = document.getElementById('recording-timer');
    this.cancelBtn = document.getElementById('cancel-recording-btn');
    this.stopBtn = document.getElementById('stop-recording-btn');
  }

  bindEvents() {
    // Panel controls
    this.chatToggle.addEventListener('click', () => this.toggleChat());
    this.closeChatBtn.addEventListener('click', () => this.closeChat());
    this.newChatBtn.addEventListener('click', () => this.newChat());

    // Input controls
    this.sendBtn.addEventListener('click', () => this.sendMessage());
    this.chatInput.addEventListener('keydown', (e) => this.handleKeyPress(e));
    this.micBtn.addEventListener('click', () => this.startRecording());

    // Recording controls
    this.cancelBtn.addEventListener('click', () => this.cancelRecording());
    this.stopBtn.addEventListener('click', () => this.stopRecording());

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this.handleGlobalKeyPress(e));
  }

  toggleChat() {
    if (this.chatWrapper.classList.contains('active')) {
      this.closeChat();
    } else {
      this.openChat();
    }
  }

  openChat() {
    this.chatWrapper.classList.add('active');
    this.chatInput.focus();
  }

  closeChat() {
    this.chatWrapper.classList.remove('active');
    this.cancelRecording(); // Cancel any ongoing recording
  }

  newChat() {
    this.chatBody.innerHTML = `
      <div class="message bot">
        <img src="nadra_logo.png" alt="NADRA Agent" class="avatar" />
        <div class="text">Hello! I'm NADRA AI AGENT, How may I assist you today? You may type your question or record it below.</div>
      </div>
    `;
    this.chatInput.value = '';
    this.cancelRecording();
  }

  handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  handleGlobalKeyPress(event) {
    if (event.key === 'Escape') {
      if (this.isRecording()) {
        this.cancelRecording();
      } else if (this.chatWrapper.classList.contains('active')) {
        this.closeChat();
      }
    }
  }

  async sendMessage() {
    const message = this.chatInput.value.trim();
    if (!message) return;

    this.chatInput.value = '';
    this.addUserMessage(message);
    this.setInputDisabled(true);

    try {
      const response = await this.queryAPI(message);
      this.addBotMessage(response);
    } catch (error) {
      console.error('Query failed:', error);
      this.addBotMessage('Sorry, I encountered an error. Please try again.');
    } finally {
      this.setInputDisabled(false);
      this.chatInput.focus();
    }
  }

  async queryAPI(text) {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.answer || 'No response received.';
  }

  addUserMessage(text) {
    const messageEl = this.createMessageElement('user', text);
    this.chatBody.appendChild(messageEl);
    this.scrollToBottom();
  }

  addBotMessage(text) {
    const messageEl = this.createMessageElement('bot', text);
    this.chatBody.appendChild(messageEl);
    this.scrollToBottom();
  }

  addUserAudioMessage(audioBlob) {
    const audioUrl = URL.createObjectURL(audioBlob);
    const messageEl = this.createAudioMessageElement('user', audioUrl);
    this.chatBody.appendChild(messageEl);
    this.scrollToBottom();
    return messageEl;
  }

  createMessageElement(type, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = document.createElement('img');
    avatar.className = 'avatar';
    avatar.alt = type === 'user' ? 'You' : 'NADRA Agent';
    avatar.src = type === 'user' ? 'user.png' : 'nadra_logo.png';

    const textDiv = document.createElement('div');
    textDiv.className = 'text';
    textDiv.textContent = text;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(textDiv);

    return messageDiv;
  }

  createAudioMessageElement(type, audioUrl) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = document.createElement('img');
    avatar.className = 'avatar';
    avatar.alt = 'You';
    avatar.src = 'user.png';

    const textDiv = document.createElement('div');
    textDiv.className = 'text';

    const audio = document.createElement('audio');
    audio.controls = true;
    audio.src = audioUrl;

    const transcription = document.createElement('div');
    transcription.className = 'transcription';
    transcription.textContent = 'Transcribing...';

    textDiv.appendChild(audio);
    textDiv.appendChild(transcription);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(textDiv);

    return messageDiv;
  }

  updateTranscription(messageEl, transcript) {
    const transcriptionEl = messageEl.querySelector('.transcription');
    if (transcriptionEl) {
      transcriptionEl.textContent = transcript ? `"${transcript}"` : 'Transcription failed';
      transcriptionEl.style.display = 'block';
      transcriptionEl.style.marginTop = '6px';
    }
  }

  scrollToBottom() {
    this.chatBody.scrollTop = this.chatBody.scrollHeight;
  }

  setInputDisabled(disabled) {
    this.chatInput.disabled = disabled;
    this.sendBtn.disabled = disabled;
    this.micBtn.disabled = disabled;
    
    if (disabled) {
      this.sendBtn.style.opacity = '0.6';
    } else {
      this.sendBtn.style.opacity = '1';
    }
  }

  // ===== RECORDING FUNCTIONALITY =====

  async startRecording() {
    if (this.isRecording()) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });

      this.recordingChunks = [];
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.recordingChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
        this.processRecording();
      };

      this.showRecordingUI();
      this.mediaRecorder.start(250); // Collect data every 250ms
      this.startRecordingTimer();

    } catch (error) {
      console.error('Recording failed:', error);
      this.addBotMessage('Unable to access microphone. Please check your permissions.');
    }
  }

  stopRecording() {
    if (this.isRecording()) {
      this.mediaRecorder.stop();
    }
    this.hideRecordingUI();
  }

  cancelRecording() {
    if (this.isRecording()) {
      this.mediaRecorder.stop();
      this.recordingChunks = []; // Clear chunks to prevent processing
    }
    this.hideRecordingUI();
  }

  isRecording() {
    return this.mediaRecorder && this.mediaRecorder.state === 'recording';
  }

  showRecordingUI() {
    this.inputArea.style.display = 'none';
    this.recordingArea.style.display = 'flex';
  }

  hideRecordingUI() {
    this.inputArea.style.display = 'flex';
    this.recordingArea.style.display = 'none';
    this.stopRecordingTimer();
  }

  startRecordingTimer() {
    this.recordingStartTime = Date.now();
    this.recordingTimer.textContent = '00:00';
    
    this.recordingTimerInterval = setInterval(() => {
      const elapsed = Date.now() - this.recordingStartTime;
      const seconds = Math.floor(elapsed / 1000);
      const minutes = Math.floor(seconds / 60);
      const displaySeconds = seconds % 60;
      
      this.recordingTimer.textContent = 
        `${minutes.toString().padStart(2, '0')}:${displaySeconds.toString().padStart(2, '0')}`;
      
      // Auto-stop at max time
      if (elapsed >= this.maxRecordingTime) {
        this.stopRecording();
      }
    }, 100);
  }

  stopRecordingTimer() {
    if (this.recordingTimerInterval) {
      clearInterval(this.recordingTimerInterval);
      this.recordingTimerInterval = null;
    }
  }

  async processRecording() {
    if (this.recordingChunks.length === 0) {
      return; // Recording was cancelled
    }

    const audioBlob = new Blob(this.recordingChunks, { type: 'audio/webm' });
    const messageEl = this.addUserAudioMessage(audioBlob);
    this.setInputDisabled(true);

    try {
      // Step 1: Transcribe
      const transcript = await this.transcribeAudio(audioBlob);
      
      if (!transcript) {
        this.updateTranscription(messageEl, null);
        this.addBotMessage("I couldn't detect any speech in your recording. Please try again.");
        return;
      }

      this.updateTranscription(messageEl, transcript);

      // Step 2: Query with transcript
      const response = await this.queryAPI(transcript);
      this.addBotMessage(response);

    } catch (error) {
      console.error('Audio processing failed:', error);
      this.updateTranscription(messageEl, null);
      this.addBotMessage("Sorry, I couldn't process your voice message. Please try typing instead.");
    } finally {
      this.setInputDisabled(false);
    }
  }

  async transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    const response = await fetch('/api/transcribe', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Transcription failed: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.error) {
      throw new Error(data.error);
    }

    return data.text?.trim() || null;
  }
}

// Initialize the chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.nadraChat = new NADRAChat();
  console.log('NADRA Agent initialized');
});