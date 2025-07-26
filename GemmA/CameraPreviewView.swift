import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager
    
    class PreviewView: UIView {
        var previewLayer: AVCaptureVideoPreviewLayer?
        
        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = bounds
        }
    }
    
    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: cameraManager.captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        view.previewLayer = previewLayer
        
        return view
    }
    
    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer?.frame = uiView.bounds
    }
} 