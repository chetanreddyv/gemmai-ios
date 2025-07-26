import AVFoundation
import Speech

class SpeechManager: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechManager()
    private let synthesizer = AVSpeechSynthesizer()
    private var isSpeaking: Bool = false
    
    // Speech Recognition Properties
    private let audioEngine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var isRecognizing = false
    var onRecognition: ((String) -> Void)?
    @Published var lastRecognizedText: String = ""
    
    private override init() {
        super.init()
        synthesizer.delegate = self
        setupAudioSession()
    }
    
    // MARK: - TTS API
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        
        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        
        isSpeaking = true
        synthesizer.speak(utterance)
    }
    
    func speakStreaming(_ text: String, isFinal: Bool = false) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        if isFinal {
            speak(trimmed)
        }
    }
    
    private func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playback, mode: .default, options: [.duckOthers, .interruptSpokenAudioAndMixWithOthers])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
    }
    
    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
    }
    
    // MARK: - AVSpeechSynthesizerDelegate
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        isSpeaking = false
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        isSpeaking = false
    }
    
    // MARK: - Speech Recognition (STT)
    func startRecognition() {
        guard !isRecognizing else { return }
        stop()
        SFSpeechRecognizer.requestAuthorization { [weak self] authStatus in
            guard let self = self else { return }
            guard authStatus == .authorized else { return }
            AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
                guard let self = self else { return }
                guard granted else { return }
                DispatchQueue.main.async {
                    self._startRecognitionSession()
                }
            }
        }
    }
    
    private func _startRecognitionSession() {
        if audioEngine.isRunning {
            audioEngine.stop()
            recognitionRequest?.endAudio()
        }
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            return
        }
        _startRecognitionEngine()
    }
    
    private func _startRecognitionEngine() {
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }
        let inputNode = audioEngine.inputNode
        recognitionRequest.shouldReportPartialResults = true
        isRecognizing = true
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            if let error = error {
                self.stopRecognition()
                return
            }
            if let result = result {
                let text = result.bestTranscription.formattedString
                self.lastRecognizedText = text
                self.onRecognition?(text)
            }
            if result?.isFinal == true {
                self.stopRecognition()
            }
        }
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            self.recognitionRequest?.append(buffer)
        }
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            self.stopRecognition()
        }
    }
    
    func stopRecognition() {
        guard isRecognizing else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionRequest = nil
        recognitionTask = nil
        isRecognizing = false
    }
    
    var isSTTRunning: Bool {
        return isRecognizing
    }
    
    var isTTSRunning: Bool {
        return isSpeaking
    }
    
    // MARK: - Audio Feedback
    private static var thinkingPlayer: AVAudioPlayer?
    
    static func playThinkingSound() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playback, mode: .default, options: [.duckOthers, .interruptSpokenAudioAndMixWithOthers])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            return
        }
        guard let url = Bundle.main.url(forResource: "thinking", withExtension: "mp3") else {
            return
        }
        do {
            thinkingPlayer = try AVAudioPlayer(contentsOf: url)
            thinkingPlayer?.prepareToPlay()
            thinkingPlayer?.play()
        } catch {
            return
        }
    }
} 

