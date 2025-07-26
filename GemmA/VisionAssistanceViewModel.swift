import Foundation
import SwiftUI
import AVFoundation
import CoreImage
import Metal

// MARK: - Constants
struct ImageQualityThresholds {
    static let sharpness: Double = 10.0
    static let brightness: Double = 0.15
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
            if let text = currentAlert, 
               text.trimmingCharacters(in: .whitespacesAndNewlines).localizedCaseInsensitiveCompare("CLEAR") != .orderedSame {
                // TTS is handled by speakStreaming method
            }
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
    private var frameBuffer: [(image: UIImage, variance: Double, brightness: Double, hash: Int, phash: UInt64)] = []
    private let bufferSize = 4
    private var bufferTimer: Timer?
    
    // Scene change detection
    private var lastDescribedScenePHash: UInt64? = nil
    private let sceneChangeThreshold: Int = 10
    private var isInSceneChangeState: Bool = false
    
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
                          topK: 20, 
                          topP: 0.8, 
                          temperature: 0.5, 
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
                try? await Task.sleep(nanoseconds: 100_000_000)
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
    func cameraManager(_ manager: CameraManager, didOutputFrame image: UIImage) {
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
        
        // Scene change detection for passive mode
        if mode == .passive {
            if let lastScenePHash = lastDescribedScenePHash {
                let hamming = hammingDistance(phash, lastScenePHash)
                if hamming < sceneChangeThreshold {
                    return
                } else {
                    isInSceneChangeState = true
                }
            }
        }
        
        frameBuffer.append((correctedImage, variance, brightness, hash, phash))
        if frameBuffer.count > bufferSize {
            frameBuffer.removeFirst()
        }
    }
    
    // MARK: - Passive Mode Processing
    private func startBufferTimer() {
        bufferTimer?.invalidate()
        bufferTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
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
        guard !isInferenceRunning, !frameBuffer.isEmpty else { 
            frameBuffer.removeAll()
            return 
        }
        
        let filtered = frameBuffer.filter { 
            $0.variance > ImageQualityThresholds.sharpness && 
            $0.brightness > ImageQualityThresholds.brightness 
        }
        
        let candidate: (image: UIImage, variance: Double, brightness: Double, hash: Int, phash: UInt64)?
        
        if isInSceneChangeState {
            candidate = filtered.last ?? frameBuffer.last
        } else {
            candidate = filtered.max(by: { $0.variance < $1.variance }) ?? 
                       frameBuffer.max(by: { $0.variance < $1.variance })
        }
        
        frameBuffer.removeAll()
        guard let selectedFrame = candidate else {
            return
        }
        
        if let lastHash = lastInferredFrameHash, lastHash == selectedFrame.hash {
            return
        }
        
        if isInSceneChangeState {
            isInSceneChangeState = false
        }
        
        lastInferredFrameHash = selectedFrame.hash
        runPassiveInference(on: selectedFrame.image, phash: selectedFrame.phash)
    }
    
    private func runPassiveInference(on image: UIImage, phash: UInt64? = nil) {
        if sessionState != .ready || isInferenceRunning {
            pendingInferenceQueue.append(InferenceRequest(prompt: nil, image: image, isActive: false))
            return
        }
        
        let imageValidation = validateImage(image)
        if !imageValidation.isValid {
            return
        }
        
        isInferenceRunning = true
        sessionState = .busy
        checkForStuckBusyState()
        
        Task {
            await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    visionProcessor?.processFramesStreaming([image]) { [weak self] partialText, isFinal in
                        DispatchQueue.main.async {
                            guard let self = self else { return }
                            
                            if partialText.contains("Error:") || partialText.contains("Processing error occurred") {
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
                                self.handleSessionError(NSError(domain: "VisionProcessor", code: -1, userInfo: [NSLocalizedDescriptionKey: partialText]))
                                continuation.resume()
                                return
                            }
                            
                            if self.mode == .passive, 
                               partialText.trimmingCharacters(in: .whitespacesAndNewlines).localizedCaseInsensitiveCompare("CLEAR") != .orderedSame {
                                self.currentAlert = partialText
                                SpeechManager.shared.speakStreaming(partialText, isFinal: isFinal)
                            }
                            
                            if isFinal {
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
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
        
        isInferenceRunning = true
        sessionState = .busy
        checkForStuckBusyState()
        mode = .activeBusy
        
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
                    visionProcessor.processFramesStreaming([image], prompt: fullPrompt) { [weak self] partialText, isFinal in
                        DispatchQueue.main.async {
                            guard let self = self else { return }
                            
                            if partialText.contains("Error:") || partialText.contains("Processing error occurred") {
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
                                self.handleSessionError(NSError(domain: "VisionProcessor", code: -1, userInfo: [NSLocalizedDescriptionKey: partialText]))
                                self.finishActiveMode()
                                continuation.resume()
                                return
                            }
                            
                            if self.mode == .activeBusy, 
                               partialText.trimmingCharacters(in: .whitespacesAndNewlines).localizedCaseInsensitiveCompare("CLEAR") != .orderedSame {
                                self.currentAlert = partialText
                                SpeechManager.shared.speakStreaming(partialText, isFinal: isFinal)
                            }
                            
                            if isFinal {
                                self.isInferenceRunning = false
                                self.sessionState = .ready
                                self.checkForStuckBusyState()
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
                    self.finishActiveMode()
                }
            }
        }
    }
    
    // MARK: - Queue Processing
    private func processInferenceQueue() {
        if let activeReq = pendingActiveRequest, !isInferenceRunning, sessionState == .ready {
            runActiveInferenceInternal(prompt: activeReq.prompt ?? "", image: activeReq.image)
            pendingActiveRequest = nil
            return
        }
        
        if let activeReq = pendingActiveRequest, sessionState != .ready {
            return
        }
        
        guard !pendingInferenceQueue.isEmpty else { 
            return 
        }
        guard !isInferenceRunning else { 
            return 
        }
        guard sessionState == .ready else { 
            return 
        }
        
        let passiveRequests = pendingInferenceQueue.filter { !$0.isActive }
        for request in passiveRequests {
            pendingInferenceQueue.removeAll { $0.image == request.image && $0.isActive == false }
            runPassiveInference(on: request.image)
            return
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
