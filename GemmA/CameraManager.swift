import AVFoundation
import UIKit

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didOutputFrame image: UIImage)
}

class CameraManager: NSObject, ObservableObject {
    @Published var isSessionRunning = false
    @Published var error: String?
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    private let videoDataQueue = DispatchQueue(label: "videoDataQueue")
    
    private var lastFrameTime: TimeInterval = 0
    private let targetFrameRate: TimeInterval = 1.0 / 10.0
    
    private var latestFrame: UIImage? = nil
    func getLatestFrame() -> UIImage? { return latestFrame }
    
    weak var delegate: CameraManagerDelegate?
    
    override init() {
        super.init()
        setupCamera()
    }
    
    private func setupCamera() {
        sessionQueue.async { [weak self] in
            self?.configureSession()
        }
    }
    
    private func checkCameraPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                completion(granted)
            }
        case .denied, .restricted:
            DispatchQueue.main.async {
                self.error = "Camera access denied. Please enable camera access in Settings."
            }
            completion(false)
        @unknown default:
            completion(false)
        }
    }
    
    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .high
        
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            DispatchQueue.main.async {
                self.error = "Failed to access camera device"
            }
            session.commitConfiguration()
            return
        }
        
        do {
            let videoInput = try AVCaptureDeviceInput(device: videoDevice)
            if session.canAddInput(videoInput) {
                session.addInput(videoInput)
            } else {
                DispatchQueue.main.async {
                    self.error = "Failed to add video input"
                }
                session.commitConfiguration()
                return
            }
        } catch {
            DispatchQueue.main.async {
                self.error = "Failed to create video input: \(error.localizedDescription)"
            }
            session.commitConfiguration()
            return
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: videoDataQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        } else {
            DispatchQueue.main.async {
                self.error = "Failed to add video output"
            }
            session.commitConfiguration()
            return
        }
        session.commitConfiguration()
        DispatchQueue.main.async {
            self.error = nil
        }
    }
    
    func startSession() {
        checkCameraPermission { [weak self] granted in
            guard let self = self, granted else { return }
            self.sessionQueue.async {
                if !self.session.isRunning {
                    self.session.startRunning()
                    DispatchQueue.main.async {
                        self.isSessionRunning = true
                    }
                }
            }
        }
    }
    
    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
                DispatchQueue.main.async {
                    self.isSessionRunning = false
                }
            }
        }
    }
    
    var captureSession: AVCaptureSession {
        return session
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastFrameTime >= targetFrameRate else { return }
        lastFrameTime = currentTime
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        
        latestFrame = uiImage
        delegate?.cameraManager(self, didOutputFrame: uiImage)
    }
} 