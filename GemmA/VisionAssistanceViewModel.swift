import Foundation
import SwiftUI
import AVFoundation
import CoreImage

// MARK: - Constants
struct ImageQualityThresholds {
    static let sharpness: Double = 15.0  // Minimum sharpness for capture AND processing
    static let brightness: Double = 0.15
    static let motionBlurThreshold: Double = 25.0  // Higher threshold for motion blur detection
    static let minimumSharpness: Double = 15.0  // Match sharpness threshold
}

// MARK: - Motion Detection
struct MotionQualityMetrics {
    let sharpness: Double
    let brightness: Double
    let isMotionBlur: Bool
    let isStable: Bool
    
    init(sharpness: Double, brightness: Double) {
        self.sharpness = sharpness
        self.brightness = brightness
        self.isMotionBlur = sharpness < ImageQualityThresholds.motionBlurThreshold
        self.isStable = sharpness > ImageQualityThresholds.minimumSharpness
    }
}

enum SessionState {
    case idle
    case initializing
    case ready
    case busy
    case error
    case recovering
}

class VisionAssistanceViewModel: ObservableObject, CameraManagerDelegate {
    enum Mode { case passive, active, activeBusy }
    
    // MARK: - Published Properties
    @Published var mode: Mode = .passive
    @Published var isModelLoading = true
    @Published var isVisionActive = false
    @Published var currentAlert: String? {
        didSet {
            // TTS is handled by speakStreaming method
        }
    }
    @Published var criticalError: String?
    @Published var activeSTTString: String = ""
    @Published var showActiveBusy: Bool = false
    @Published var sessionState: SessionState = .idle
    
    // MARK: - Private Properties
    var cameraManager: CameraManager? { _cameraManager }
    private var _cameraManager: CameraManager?
    private var visionProcessor: VisionProcessor?
    private var chat: Chat?
    private var onDeviceModel: OnDeviceModel?
    private let inferenceQueue = DispatchQueue(label: "inferenceQueue", qos: .userInitiated)
    @Published var isInferenceRunning = false
    private var pendingFrame: UIImage? = nil
    private var lastInferredFrameHash: Int? = nil
    private let inferenceTimeout: TimeInterval = 15.0
    private var inferenceWatchdog: Timer?
    private var frameBuffer: [(image: UIImage, variance: Double, brightness: Double, hash: Int, phash: UInt64, metrics: MotionQualityMetrics)] = []
    private let bufferSize = 4  // 4 frames per second as requested
    private var bufferTimer: Timer?
    
    // Scene change detection
    private var lastDescribedScenePHash: UInt64? = nil
    private let sceneChangeThreshold: Int = 10
    private var isInSceneChangeState: Bool = false
    
    // Motion detection
    private var recentFrameHashes: [Int] = []
    private let maxRecentHashes = 5
    private var isHighMotion: Bool = false
    
    // MARK: - Task Management
    private var passivePaused: Bool = false
    private var lastActiveBusyTime: Date? = nil
    
    // MARK: - Queue Management
    private var pendingInferenceQueue: [InferenceRequest] = []
    private var pendingActiveRequest: InferenceRequest? = nil
    
    struct InferenceRequest {
        let prompt: String?
        let image: UIImage
        let isActive: Bool
    }
    
    // MARK: - Session Management
    private var currentSessionId: UUID?
    private var sessionErrorCount: Int = 0
    private let maxSessionErrors = 3
    private var lastSessionReset: Date?
    private let sessionResetCooldown: TimeInterval = 5.0
    private var busyStateStartTime: Date?
    private let busyStateTimeout: TimeInterval = 30.0
    
    // MARK: - Input Validation
    private let minPromptLength = 1
    private let maxPromptLength = 512
    private let minImageSize = CGSize(width: 100, height: 100)
    
    // MARK: - Debug and Logging
    private var lastInferenceStartTime: Date?
    private var inferenceTimeoutTimer: Timer?
    
    // MARK: - Mode Management
    private var currentMode: Mode = .passive
    private var isModeBusy: Bool = false
    
    init() {
        Task {
            await initializeVisionSystem()
        }
        startBufferTimer()
    }
    
    // MARK: - Initialization
    @MainActor
    func initializeVisionSystem() async {
        self.isModelLoading = true
        self.criticalError = nil
        self.sessionState = .initializing
        
        do {
            onDeviceModel = try OnDeviceModel()
            
            chat = try Chat(model: onDeviceModel!, 
                          enableVisionModality: true)
            
            visionProcessor = VisionProcessor(chat: chat!)
            
            _cameraManager = CameraManager()
            _cameraManager?.delegate = self
            
            self.isModelLoading = false
            self.sessionState = .ready
            self.currentSessionId = UUID()
            
            _cameraManager?.startSession()
            self.isVisionActive = true
            
        } catch {
            self.criticalError = "Failed to initialize vision system: \(error.localizedDescription)"
            self.isModelLoading = false
            self.sessionState = .error
        }
    }
    
    // MARK: - Session Management
    private func resetSession() async {
        let now = Date()
        if let lastReset = lastSessionReset, now.timeIntervalSince(lastReset) < sessionResetCooldown {
            return
        }
        
        sessionState = .recovering
        sessionErrorCount = 0
        lastSessionReset = now
        isInferenceRunning = false
        
        if SpeechManager.shared.isTTSRunning || SpeechManager.shared.isSTTRunning {
            SpeechManager.shared.stop()
        }
        
        isInferenceRunning = false
        frameBuffer.removeAll()
        pendingInferenceQueue.removeAll()
        pendingFrame = nil
        lastInferredFrameHash = nil
        lastDescribedScenePHash = nil
        
        await initializeVisionSystem()
        
        if sessionState == .ready {
            if let activeReq = pendingActiveRequest {
                try? await Task.sleep(nanoseconds: 200_000_000)
                runActiveInferenceInternal(prompt: activeReq.prompt ?? "", image: activeReq.image, retryingAfterReset: true)
                pendingActiveRequest = nil
            } else {
                processInferenceQueue()
            }
        }
    }
    
    private func handleSessionError(_ error: Error) {
        sessionErrorCount += 1
        
        let isContextError = error.localizedDescription.contains("OUT_OF_RANGE") || 
                            error.localizedDescription.contains("exceed context window") ||
                            error.localizedDescription.contains("current_step") ||
                            error.localizedDescription.contains("Processing error occurred")
        
        if sessionErrorCount >= maxSessionErrors || isContextError {
            Task {
                await resetSession()
            }
        }
    }
    
    // MARK: - Input Validation
    private func validatePrompt(_ prompt: String) -> (isValid: Bool, error: String?) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        
        if trimmed.isEmpty {
            return (false, "Prompt is empty")
        }
        
        if trimmed.count < minPromptLength {
            return (false, "Prompt too short (minimum \(minPromptLength) characters)")
        }
        
        if trimmed.count > maxPromptLength {
            return (false, "Prompt too long (maximum \(maxPromptLength) characters)")
        }
        
        return (true, nil)
    }
    
    private func validateImage(_ image: UIImage) -> (isValid: Bool, error: String?) {
        guard let cgImage = image.cgImage else {
            return (false, "Invalid image data")
        }
        
        let size = CGSize(width: cgImage.width, height: cgImage.height)
        if size.width < minImageSize.width || size.height < minImageSize.height {
            return (false, "Image too small (minimum \(Int(minImageSize.width))x\(Int(minImageSize.height)))")
        }
        
        return (true, nil)
    }
    
    // MARK: - Mode Management
    func pausePassiveMode() {
        passivePaused = true
        SpeechManager.shared.stop()
        DispatchQueue.main.async { self.currentAlert = nil }
        isInferenceRunning = false
        frameBuffer.removeAll()
    }
    
    func resumePassiveMode() {
        passivePaused = false
    }
    
    func startActiveMode() {
        pausePassiveMode()
        mode = .active
        activeSTTString = ""
        showActiveBusy = false
    }
    
    func finishActiveMode() {
        mode = .passive
        showActiveBusy = false
        resumePassiveMode()
    }
    
    // MARK: - Camera Management
    private var lastFrameCaptureTime: Date?
    private let frameCaptureDelay: TimeInterval = 1.0  // 1 second delay between frame captures
    
    func cameraManager(_ manager: CameraManager, didOutputFrame image: UIImage) {
        // Add delay to frame capturing for stability
        if let lastCapture = lastFrameCaptureTime,
           Date().timeIntervalSince(lastCapture) < frameCaptureDelay {
            return  // Skip this frame, wait for delay
        }
        
        let correctedImage = image.fixedOrientation()
        let variance = correctedImage.laplacianVariance()
        let brightness = correctedImage.meanBrightness()
        let hash = correctedImage.fastHash()
        let phash = correctedImage.perceptualHash64()
        
        // Quality checks
        if variance <= ImageQualityThresholds.sharpness {
            return
        }
        if brightness <= ImageQualityThresholds.brightness {
            return
        }
        if let lastHash = lastInferredFrameHash, lastHash == hash {
            return
        }
        if isInferenceRunning {
            return
        }
        
        // Add frame to buffer (scene change detection moved to buffer processing)
        let metrics = MotionQualityMetrics(sharpness: variance, brightness: brightness)
        frameBuffer.append((correctedImage, variance, brightness, hash, phash, metrics))
        lastFrameCaptureTime = Date()  // Update capture time
        
        // Track motion by monitoring frame hash changes
        recentFrameHashes.append(hash)
        if recentFrameHashes.count > maxRecentHashes {
            recentFrameHashes.removeFirst()
        }
        
        // Detect high motion if we have multiple different hashes recently
        let uniqueHashes = Set(recentFrameHashes)
        isHighMotion = uniqueHashes.count >= 3
        
        print("🔍 ViewModel: Frame added - Sharpness: \(variance), Motion blur: \(metrics.isMotionBlur), Stable: \(metrics.isStable), High motion: \(isHighMotion)")
        
        // Keep buffer size limited - keep last 4 frames instead of first 4
        if frameBuffer.count > bufferSize {
            frameBuffer.removeFirst()  // Remove oldest frame, keep newest 4
        }
    }
    
    // MARK: - Passive Mode Processing
    private func startBufferTimer() {
        bufferTimer?.invalidate()
        let interval = frameCaptureDelay  // Match the frame capture delay (1.0s)
        bufferTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.processFrameBuffer()
            self?.processInferenceQueue()
            self?.checkForStuckBusyState()
        }
    }
    
    private func processFrameBuffer() {
        guard mode == .passive, !passivePaused else { 
            frameBuffer.removeAll()
            return 
        }
        guard !isInferenceRunning else { 
            return 
        }
        guard !frameBuffer.isEmpty else { 
            return 
        }
        
        print("🔍 ViewModel: Processing frame buffer, size: \(frameBuffer.count)")
        
        // Scene change detection at buffer processing level
        if mode == .passive {
            let latestFrame = frameBuffer.last
            if let latestPHash = latestFrame?.phash, let lastScenePHash = lastDescribedScenePHash {
                let hamming = hammingDistance(latestPHash, lastScenePHash)
                if hamming >= sceneChangeThreshold {
                    print("🔍 ViewModel: Scene change detected (hamming: \(hamming) >= \(sceneChangeThreshold))")
                    isInSceneChangeState = true
                } else {
                    print("🔍 ViewModel: Similar scene (hamming: \(hamming) < \(sceneChangeThreshold))")
                }
            }
        }
        
        let filtered = frameBuffer.filter { 
            $0.metrics.isStable && 
            $0.brightness > ImageQualityThresholds.brightness 
        }
        
        print("🔍 ViewModel: Filtered frames: \(filtered.count)/\(frameBuffer.count)")
        
        // If no frames pass quality filter, use the best available
        let candidate: (image: UIImage, variance: Double, brightness: Double, hash: Int, phash: UInt64, metrics: MotionQualityMetrics)?
        
        if filtered.isEmpty {
            print("⚠️ ViewModel: No frames passed quality filter, using best available")
            candidate = frameBuffer.max(by: { $0.variance < $1.variance })
        } else if isInSceneChangeState {
            // During scene change, prefer the most recent stable frame
            candidate = filtered.last ?? frameBuffer.filter { $0.metrics.isStable }.last ?? frameBuffer.last
            print("🔍 ViewModel: Scene change detected, using last stable frame")
        } else {
            // For similar scenes, prefer the most recent frame to avoid processing old data
            candidate = filtered.last ?? frameBuffer.last
            print("🔍 ViewModel: Similar scene, using most recent frame")
        }
        
        // Clear buffer before processing to prevent accumulation
        frameBuffer.removeAll()
        
        guard let selectedFrame = candidate else {
            print("❌ ViewModel: No candidate frame found")
            return
        }
        
        if let lastHash = lastInferredFrameHash, lastHash == selectedFrame.hash {
            print("🔍 ViewModel: Rejecting duplicate frame (hash: \(selectedFrame.hash))")
            return
        }
        
        if isInSceneChangeState {
            isInSceneChangeState = false
            print("🔍 ViewModel: Resetting scene change state")
        }
        
        lastInferredFrameHash = selectedFrame.hash
        print("✅ ViewModel: Selected frame for inference, hash: \(selectedFrame.hash)")
        print("📊 ViewModel: Frame metrics - Sharpness: \(selectedFrame.metrics.sharpness), Motion blur: \(selectedFrame.metrics.isMotionBlur), Stable: \(selectedFrame.metrics.isStable)")
        runPassiveInference(on: selectedFrame.image, phash: selectedFrame.phash)
    }
    
    private func runPassiveInference(on image: UIImage, phash: UInt64? = nil) {
        // Check if we can run passive mode
        guard currentMode == .passive && !isModeBusy else {
            print("⚠️ ViewModel: Cannot run passive inference - current mode: \(currentMode), mode busy: \(isModeBusy)")
            pendingInferenceQueue.append(InferenceRequest(prompt: nil, image: image, isActive: false))
            return
        }
        
        if sessionState != .ready || isInferenceRunning {
            print("⚠️ ViewModel: Cannot start inference - session: \(sessionState), running: \(isInferenceRunning)")
            pendingInferenceQueue.append(InferenceRequest(prompt: nil, image: image, isActive: false))
            return
        }
        
        let imageValidation = validateImage(image)
        if !imageValidation.isValid {
            print("❌ ViewModel: Image validation failed: \(imageValidation.error ?? "unknown")")
            return
        }
        
        let imageHash = image.fastHash()
        print("🔍 ViewModel: Starting passive inference for image hash: \(imageHash)")
        
        isModeBusy = true
        isInferenceRunning = true
        sessionState = .busy
        lastInferenceStartTime = Date()
        checkForStuckBusyState()
        
        // Start timeout timer
        startInferenceTimeout()
        
        Task {
            await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    var hasReceivedAnyOutput = false
                    var finalOutput = ""
                    
                    visionProcessor?.processFramesStreaming([image]) { [weak self] partialText, isFinal in
                        DispatchQueue.main.async {
                            guard let self = self else { return }
                            
                            if partialText.contains("Error:") || partialText.contains("Processing error occurred") {
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
                                self.stopInferenceTimeout()
                                self.isModeBusy = false  // Reset mode busy flag
                                self.handleSessionError(NSError(domain: "VisionProcessor", code: -1, userInfo: [NSLocalizedDescriptionKey: partialText]))
                                continuation.resume()
                                return
                            }
                            
                            // Track if we've received any meaningful output
                            let trimmedText = partialText.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmedText.isEmpty {
                                hasReceivedAnyOutput = true
                                finalOutput = partialText
                            }
                            
                            if self.mode == .passive {
                                self.currentAlert = partialText
                                SpeechManager.shared.speakStreaming(partialText, isFinal: isFinal)
                            }
                            
                            if isFinal {
                                // Check if we got no meaningful output
                                if !hasReceivedAnyOutput {
                                    self.currentAlert = "No response generated"
                                    SpeechManager.shared.speak("No response generated")
                                }
                                
                                print("✅ ViewModel: Passive inference completed, response: '\(finalOutput)'")
                                print("🔍 ViewModel: Image hash: \(imageHash)")
                                
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
                                self.stopInferenceTimeout()
                                self.isModeBusy = false  // Reset mode busy flag
                                if let phash = phash {
                                    self.lastDescribedScenePHash = phash
                                }
                                continuation.resume()
                                self.processInferenceQueue()
                            }
                        }
                    }
                }
                            } onCancel: {
                    DispatchQueue.main.async {
                        [weak self] in
                        guard let self = self else { return }
                        self.isInferenceRunning = false
                        self.sessionState = .ready
                        self.checkForStuckBusyState()
                        self.stopInferenceTimeout()
                        self.isModeBusy = false  // Reset mode busy flag
                    }
                }
        }
    }
    
    // MARK: - Active Mode Processing
    @MainActor
    func runActiveModeInference(with prompt: String) {
        if isInferenceRunning {
            if let latestFrame = cameraManager?.getLatestFrame() {
                let correctedImage = latestFrame.fixedOrientation()
                pendingActiveRequest = InferenceRequest(prompt: prompt, image: correctedImage, isActive: true)
            } else {
                currentAlert = "No camera frame available"
                finishActiveMode()
            }
            return
        }
        
        if mode == .activeBusy {
            showActiveBusy = true
            lastActiveBusyTime = Date()
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 500_000_000)
                DispatchQueue.main.async { self?.showActiveBusy = false }
            }
            return
        }
        
        let promptValidation = validatePrompt(prompt)
        if !promptValidation.isValid {
            currentAlert = "Invalid prompt: \(promptValidation.error ?? "unknown error")"
            finishActiveMode()
            return
        }
        
        guard let latestFrame = cameraManager?.getLatestFrame() else {
            currentAlert = "No camera frame available"
            finishActiveMode()
            return
        }
        
        let correctedImage = latestFrame.fixedOrientation()
        runActiveInferenceInternal(prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines), image: correctedImage)
    }
    
    private func runActiveInferenceInternal(prompt: String, image: UIImage, retryingAfterReset: Bool = false) {
        // Check if we can run active mode
        guard !isModeBusy else {
            print("⚠️ ViewModel: Cannot run active inference - mode is busy")
            currentAlert = "System busy, please wait"
            finishActiveMode()
            return
        }
        
        let imageValidation = validateImage(image)
        if !imageValidation.isValid {
            currentAlert = "Invalid image: \(imageValidation.error ?? "unknown error")"
            finishActiveMode()
            return
        }
        
        let variance = image.laplacianVariance()
        let brightness = image.meanBrightness()
        if variance <= ImageQualityThresholds.sharpness {
            currentAlert = "Frame not sharp enough. Try again."
            finishActiveMode()
            return
        }
        if brightness <= ImageQualityThresholds.brightness {
            currentAlert = "Frame too dark. Try again."
            finishActiveMode()
            return
        }
        
        if sessionState != .ready {
            pendingActiveRequest = InferenceRequest(prompt: prompt, image: image, isActive: true)
            return
        }
        
        isModeBusy = true
        isInferenceRunning = true
        sessionState = .busy
        lastInferenceStartTime = Date()
        checkForStuckBusyState()
        mode = .activeBusy
        
        // Start timeout timer
        startInferenceTimeout()
        
        guard let visionProcessor = self.visionProcessor else {
            currentAlert = "Model not ready"
            finishActiveMode()
            return
        }
        
        let systemPrompt = "Answer the user's question about this image in 7 words or fewer."
        let fullPrompt = "\(systemPrompt)\n\nUser: \(prompt)?"
        
        Task {
            await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    var hasReceivedAnyOutput = false
                    var finalOutput = ""
                    
                    visionProcessor.processFramesStreaming([image], prompt: fullPrompt) { [weak self] partialText, isFinal in
                        DispatchQueue.main.async {
                            guard let self = self else { return }
                            
                            if partialText.contains("Error:") || partialText.contains("Processing error occurred") {
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
                                self.stopInferenceTimeout()
                                self.isModeBusy = false  // Reset mode busy flag
                                self.handleSessionError(NSError(domain: "VisionProcessor", code: -1, userInfo: [NSLocalizedDescriptionKey: partialText]))
                                self.finishActiveMode()
                                continuation.resume()
                                return
                            }
                            
                            // Track if we've received any meaningful output
                            let trimmedText = partialText.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmedText.isEmpty {
                                hasReceivedAnyOutput = true
                                finalOutput = partialText
                            }
                            
                            if self.mode == .activeBusy {
                                self.currentAlert = partialText
                                SpeechManager.shared.speakStreaming(partialText, isFinal: isFinal)
                            }
                            
                            if isFinal {
                                // Check if we got no meaningful output
                                if !hasReceivedAnyOutput {
                                    self.currentAlert = "No response generated"
                                    SpeechManager.shared.speak("No response generated")
                                }
                                
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
                                self.stopInferenceTimeout()
                                self.isModeBusy = false  // Reset mode busy flag
                                self.finishActiveMode()
                                continuation.resume()
                                self.processInferenceQueue()
                            }
                        }
                    }
                }
                            } onCancel: {
                    DispatchQueue.main.async {
                        [weak self] in
                        guard let self = self else { return }
                        self.isInferenceRunning = false
                        self.sessionState = .ready
                        self.checkForStuckBusyState()
                        self.stopInferenceTimeout()
                        self.isModeBusy = false  // Reset mode busy flag
                        self.finishActiveMode()
                    }
                }
        }
    }
    
    // MARK: - Timeout Management
    private func startInferenceTimeout() {
        stopInferenceTimeout()
        inferenceTimeoutTimer = Timer.scheduledTimer(withTimeInterval: inferenceTimeout, repeats: false) { [weak self] _ in
            DispatchQueue.main.async {
                self?.handleInferenceTimeout()
            }
        }
    }
    
    private func stopInferenceTimeout() {
        inferenceTimeoutTimer?.invalidate()
        inferenceTimeoutTimer = nil
    }
    
    private func handleInferenceTimeout() {
        isInferenceRunning = false
        sessionState = .ready
        checkForStuckBusyState()
        
        if mode == .activeBusy {
            currentAlert = "Inference timed out"
            SpeechManager.shared.speak("Inference timed out")
            finishActiveMode()
        } else {
            currentAlert = "Inference timed out"
            SpeechManager.shared.speak("Inference timed out")
        }
        
        processInferenceQueue()
    }
    
    // MARK: - Queue Processing
    private func processInferenceQueue() {
        // Handle active requests first
        if let activeReq = pendingActiveRequest, !isInferenceRunning, sessionState == .ready {
            runActiveInferenceInternal(prompt: activeReq.prompt ?? "", image: activeReq.image)
            pendingActiveRequest = nil
            return
        }
        
        // Handle passive requests
        guard !pendingInferenceQueue.isEmpty else { 
            return 
        }
        guard !isInferenceRunning else { 
            return 
        }
        guard sessionState == .ready else { 
            return 
        }
        
        // Process the first passive request
        if let request = pendingInferenceQueue.first {
            pendingInferenceQueue.removeFirst()
            runPassiveInference(on: request.image)
        }
    }
    
    // MARK: - Task Management
    private func resetModelAndChat() {
        DispatchQueue.main.async {
            self.visionProcessor = nil
            self.chat = nil
            self.onDeviceModel = nil
            self.isModelLoading = true
            self.sessionState = .initializing
            Task {
                await self.initializeVisionSystem()
            }
        }
    }
    
    private func checkForStuckBusyState() {
        if sessionState == .busy {
            if busyStateStartTime == nil {
                busyStateStartTime = Date()
            } else if let startTime = busyStateStartTime, 
                      Date().timeIntervalSince(startTime) > busyStateTimeout {
                Task {
                    await resetSession()
                }
            }
        } else {
            busyStateStartTime = nil
        }
    }
    
    deinit {
        _cameraManager?.stopSession()
    }
    

}

// MARK: - Image Extensions
extension UIImage {
    func fastHash() -> Int {
        let size = CGSize(width: 128, height: 128)
        UIGraphicsBeginImageContext(size)
        self.draw(in: CGRect(origin: .zero, size: size))
        let smallImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let cgImage = smallImage?.cgImage else { return 0 }
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerRow = cgImage.bytesPerRow
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else { return 0 }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { return 0 }
        
        let buffer = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        var hash = 0
        
        let stepSize = max(1, width * height / 2000)
        for i in stride(from: 0, to: width * height * 4, by: stepSize * 4) {
            let r = Int(buffer[i])
            let g = Int(buffer[i + 1])
            let b = Int(buffer[i + 2])
            hash = hash &* 31 &+ r &* 37 &+ g &* 41 &+ b &* 43
        }
        
        return hash
    }
    
    func laplacianVariance() -> Double {
        guard let cgImage = self.cgImage else { return 0 }
        let ciImage = CIImage(cgImage: cgImage)
        let filter = CIFilter(name: "Laplacian") ?? CIFilter(name: "CIConvolution3X3")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)
        if let laplacian = filter?.outputImage {
            let context = CIContext()
            if let output = context.createCGImage(laplacian, from: laplacian.extent) {
                let provider = output.dataProvider
                let pixelData = provider?.data
                let length = CFDataGetLength(pixelData)
                var sum: Double = 0
                for i in 0..<length {
                    let val = CFDataGetBytePtr(pixelData)![i]
                    sum += Double(val) * Double(val)
                }
                let variance = sum / Double(length)
                return variance
            }
        }
        return 0
    }
    
    func meanBrightness() -> Double {
        guard let cgImage = self.cgImage else { return 0 }
        let ciImage = CIImage(cgImage: cgImage)
        let extent = ciImage.extent
        let context = CIContext()
        let inputExtent = CIVector(x: extent.origin.x, y: extent.origin.y, z: extent.size.width, w: extent.size.height)
        let filter = CIFilter(name: "CIAreaAverage", parameters: [kCIInputImageKey: ciImage, kCIInputExtentKey: inputExtent])
        guard let outputImage = filter?.outputImage else { return 0 }
        var bitmap = [UInt8](repeating: 0, count: 4)
        context.render(outputImage, toBitmap: &bitmap, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: nil)
        let brightness = (0.299 * Double(bitmap[0]) + 0.587 * Double(bitmap[1]) + 0.114 * Double(bitmap[2])) / 255.0
        return brightness
    }
    
    func fixedOrientation() -> UIImage {
        if imageOrientation == .up { return self }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return normalizedImage ?? self
    }
    
    func perceptualHash64() -> UInt64 {
        let size = CGSize(width: 8, height: 8)
        UIGraphicsBeginImageContext(size)
        self.draw(in: CGRect(origin: .zero, size: size))
        let smallImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        guard let cgImage = smallImage?.cgImage else { return 0 }
        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bytesPerRow = width
        var pixelData = [UInt8](repeating: 0, count: width * height)
        guard let context = CGContext(data: &pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0) else { return 0 }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        let mean = pixelData.reduce(0, { $0 + Int($1) }) / pixelData.count
        var hash: UInt64 = 0
        for (i, pixel) in pixelData.enumerated() {
            if pixel > mean {
                hash |= (1 << UInt64(i))
            }
        }
        return hash
    }
}

func hammingDistance(_ a: UInt64, _ b: UInt64) -> Int {
    return (a ^ b).nonzeroBitCount
} 
