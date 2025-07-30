import SwiftUI
import Combine

struct VisionAssistanceView: View {
    @StateObject private var viewModel = VisionAssistanceViewModel()
    @State private var showingSettings = false
    @State private var isListening = false
    @State private var isUserHolding = false
    @State private var sttCancellable: AnyCancellable? = nil
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            GeometryReader { geometry in
                if let cameraManager = viewModel.cameraManager {
                    CameraPreviewView(cameraManager: cameraManager)
                        .frame(width: min(geometry.size.width, geometry.size.height), height: min(geometry.size.width, geometry.size.height))
                        .clipped()
                        .position(x: geometry.size.width/2, y: geometry.size.height/2)
                    
                    VStack {
                        Spacer()
                        Text("Tap anywhere on screen to ask a question")
                            .font(.headline)
                            .foregroundColor(.white.opacity(0.92))
                            .padding(.vertical, 18)
                            .frame(maxWidth: .infinity)
                            .background(Color.black.opacity(0.45))
                            .cornerRadius(16)
                            .padding(.horizontal, 32)
                            .accessibilityLabel("Tap anywhere on screen to ask a question")
                    }
                    .frame(width: geometry.size.width, height: geometry.size.height, alignment: .bottom)
                } else {
                    Color.black
                        .frame(width: min(geometry.size.width, geometry.size.height), height: min(geometry.size.width, geometry.size.height))
                        .position(x: geometry.size.width/2, y: geometry.size.height/2)
                        .overlay(
                            VStack {
                                Image(systemName: "camera.fill")
                                    .font(.largeTitle)
                                    .foregroundColor(.white)
                                Text("Camera not available")
                                    .foregroundColor(.white)
                                    .padding(.top)
                            }
                        )
                }
            }
            
            VStack {
                topStatusBar
                Spacer()
            }
            
            if let alert = viewModel.currentAlert, !alert.isEmpty {
                alertOverlay(alert)
            }
            
            if viewModel.isModelLoading {
                StartupView()
            }
            
            if let error = viewModel.criticalError {
                errorOverlay(error)
            }
            
            if let cameraManager = viewModel.cameraManager, let error = cameraManager.error {
                errorOverlay(error)
            }
            
            if isUserHolding {
                ListeningOverlay()
            }
            
            if viewModel.showActiveBusy {
                ActiveBusyOverlay()
            }
            
            if viewModel.isInferenceRunning {
                ThinkingOverlay()
            }
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    if !isUserHolding && viewModel.mode == .passive {
                        isUserHolding = true
                        SpeechManager.shared.stop()
                        isListening = true
                        viewModel.startActiveMode()
                        SpeechManager.shared.startRecognition()
                        sttCancellable = SpeechManager.shared.$lastRecognizedText
                            .sink { text in
                                if viewModel.mode == .active && isListening {
                                    viewModel.activeSTTString = text
                                }
                            }
                    } else if viewModel.mode == .activeBusy {
                        viewModel.showActiveBusy = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                            viewModel.showActiveBusy = false
                        }
                    }
                }
                .onEnded { _ in
                    if isUserHolding {
                        isUserHolding = false
                        if isListening {
                            isListening = false
                            SpeechManager.shared.stopRecognition()
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                let finalPrompt = viewModel.activeSTTString.trimmingCharacters(in: .whitespacesAndNewlines)
                                if !finalPrompt.isEmpty {
                                    viewModel.activeSTTString = ""
                                    SpeechManager.playThinkingSound()
                                    viewModel.runActiveModeInference(with: finalPrompt)
                                } else {
                                    viewModel.finishActiveMode()
                                }
                                sttCancellable?.cancel()
                            }
                        }
                    }
                }
        )
        .onAppear {
            SpeechManager.shared.onRecognition = { text in
                // Speech recognition handling
            }
        }
        .sheet(isPresented: $showingSettings) {
            VisionSettingsView(viewModel: viewModel)
        }
    }
    
    // MARK: - UI Components
    private var topStatusBar: some View {
        HStack(spacing: 16) {
            HStack(spacing: 8) {
                Circle()
                    .fill(viewModel.isVisionActive ? Color.green : Color.red)
                    .frame(width: 12, height: 12)
                Text(viewModel.isVisionActive ? "Vision Active" : "Vision Inactive")
                    .font(.caption)
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.black.opacity(0.6))
            .cornerRadius(20)
            
            HStack(spacing: 8) {
                Image(systemName: viewModel.mode == .active ? "bolt.fill" : "moon.fill")
                    .foregroundColor(viewModel.mode == .active ? .green : .gray)
                Text(viewModel.mode == .active ? "Active Mode" : "Passive Mode")
                    .font(.caption2)
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.black.opacity(0.6))
            .cornerRadius(12)
            
            Spacer()
            
            Button {
                showingSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .font(.title2)
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.6))
                    .clipShape(Circle())
            }
        }
        .padding(.top, 8)
        .padding(.horizontal)
        .accessibilityElement(children: .combine)
    }
    
    private func alertOverlay(_ alert: String) -> some View {
        VStack {
            Spacer()
            
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                    .font(.title2)
                
                Text(alert)
                    .font(.headline)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.leading)
            }
            .padding()
            .background(Color.black.opacity(0.8))
            .cornerRadius(12)
            .padding(.horizontal)
            
            Spacer()
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .animation(.easeInOut(duration: 0.3), value: alert)
    }
    
    private func errorOverlay(_ error: String) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.largeTitle)
                .foregroundColor(.red)
            
            Text("Error")
                .font(.headline)
                .foregroundColor(.white)
            
            Text(error)
                .font(.body)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
            
            Button("Retry") {
                Task {
                    await viewModel.initializeVisionSystem()
                }
            }
            .foregroundColor(.blue)
            .padding(.horizontal, 24)
            .padding(.vertical, 8)
            .background(Color.white)
            .cornerRadius(8)
        }
        .padding(40)
        .background(Color.black.opacity(0.9))
        .cornerRadius(20)
        .padding()
    }
}

// MARK: - Startup View
struct BouncingDots: View {
    @State private var animate = [false, false, false]
    let dotCount = 3
    let dotSize: CGFloat = 18
    let spacing: CGFloat = 16
    let color = Color.teal
    
    var body: some View {
        HStack(spacing: spacing) {
            ForEach(0..<dotCount, id: \.self) { i in
                Circle()
                    .fill(color)
                    .frame(width: dotSize, height: dotSize)
                    .offset(y: animate[i] ? -14 : 14)
                    .animation(
                        Animation.easeInOut(duration: 0.5)
                            .repeatForever()
                            .delay(Double(i) * 0.18),
                        value: animate[i]
                    )
            }
        }
        .onAppear {
            for i in 0..<dotCount {
                DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.18) {
                    withAnimation {
                        animate[i].toggle()
                    }
                }
            }
            
            Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { _ in
                for i in 0..<dotCount {
                    DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.18) {
                        withAnimation {
                            animate[i].toggle()
                        }
                    }
                }
            }
        }
        .accessibilityHidden(true)
    }
}

struct StartupView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 36) {
                Spacer()
                Text("GemmA.I")
                    .font(.system(size: 48, weight: .black, design: .rounded))
                    .foregroundColor(.white)
                    .shadow(color: .teal.opacity(0.3), radius: 10, x: 0, y: 2)
                Text("Visual perception for the blind")
                    .font(.title3)
                    .foregroundColor(.white.opacity(0.8))
                BouncingDots()
                    .frame(height: 36)
                    .padding(.top, 8)
                VStack(spacing: 8) {
                    Text("Loading vision model and camera...")
                        .font(.body)
                        .foregroundColor(.white.opacity(0.8))
                    HStack(spacing: 6) {
                        Text("Powered by")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.7))
                        Text("Gemma 3n")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(.teal)
                            .shadow(color: .teal.opacity(0.3), radius: 4, x: 0, y: 1)
                    }
                    Text("by Chetan Valluru")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                        .padding(.top, 4)
                }
                Spacer()
            }
            .padding(.horizontal, 32)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Loading GemmA.I, visual perception for the blind, powered by Gemma 3n, by Chetan Valluru")
    }
}

// MARK: - Settings View
struct VisionSettingsView: View {
    @ObservedObject var viewModel: VisionAssistanceViewModel
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Vision Assistance")) {
                    HStack {
                        Text("Status")
                        Spacer()
                        Text(viewModel.isVisionActive ? "Active" : "Inactive")
                            .foregroundColor(viewModel.isVisionActive ? .green : .red)
                    }
                    
                    HStack {
                        Text("Model")
                        Spacer()
                        Text("Gemma 3N 2B")
                            .foregroundColor(.secondary)
                    }
                }
                

                
                Section(header: Text("About")) {
                    Text("This app uses AI vision to detect hazards and obstacles, providing audio alerts to assist with navigation.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Overlay Views
struct ListeningOverlay: View {
    var body: some View {
        Color.clear
            .background(
                GeometryReader { geometry in
                    RoundedRectangle(cornerRadius: 32, style: .continuous)
                        .fill(Color.green.opacity(0.7))
                        .frame(width: geometry.size.width * 0.95, height: geometry.size.height * 0.9)
                        .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                        .overlay(
                            Text("Listening")
                                .font(.largeTitle)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .shadow(radius: 10)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 32, style: .continuous)
                                .stroke(Color.white.opacity(0.7), lineWidth: 4)
                        )
                        .accessibilityElement(children: .combine)
                        .accessibilityLabel("Listening for speech")
                }
            )
            .edgesIgnoringSafeArea(.all)
    }
}

struct ActiveBusyOverlay: View {
    var body: some View {
        Color.clear
            .background(
                GeometryReader { geometry in
                    RoundedRectangle(cornerRadius: 32, style: .continuous)
                        .fill(Color.red.opacity(0.8))
                        .frame(width: geometry.size.width * 0.95, height: geometry.size.height * 0.9)
                        .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                        .overlay(
                            Text("Task already running")
                                .font(.largeTitle)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .shadow(radius: 10)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 32, style: .continuous)
                                .stroke(Color.white.opacity(0.7), lineWidth: 4)
                        )
                        .accessibilityElement(children: .combine)
                        .accessibilityLabel("Task already running")
                }
            )
            .edgesIgnoringSafeArea(.all)
    }
}

struct ThinkingOverlay: View {
    @State private var animate = false
    @State private var pulseScale: CGFloat = 1.0
    
    var body: some View {
        VStack {
            Spacer()
            
            HStack(spacing: 12) {
                HStack(spacing: 6) {
                    ForEach(0..<3, id: \.self) { index in
                        Circle()
                            .fill(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.blue, Color.purple]),
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 10, height: 10)
                            .scaleEffect(animate ? 1.2 : 0.8)
                            .opacity(animate ? 1.0 : 0.4)
                            .animation(
                                Animation.easeInOut(duration: 0.8)
                                    .repeatForever()
                                    .delay(Double(index) * 0.15),
                                value: animate
                            )
                            .shadow(color: .blue.opacity(0.6), radius: 2, x: 0, y: 1)
                    }
                }
                
                Text("Thinking...")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
                    .shadow(color: .black.opacity(0.8), radius: 2, x: 0, y: 1)
                    .scaleEffect(pulseScale)
                    .animation(
                        Animation.easeInOut(duration: 1.2)
                            .repeatForever(autoreverses: true),
                        value: pulseScale
                    )
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.black.opacity(0.9))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(
                                LinearGradient(
                                    gradient: Gradient(colors: [Color.blue.opacity(0.8), Color.purple.opacity(0.6)]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ),
                                lineWidth: 1.5
                            )
                    )
            )
            .shadow(color: .black.opacity(0.6), radius: 8, x: 0, y: 4)
            .scaleEffect(pulseScale * 0.98)
            
            Spacer()
                .frame(height: 80)
        }
        .onAppear {
            animate = true
            pulseScale = 1.05
        }
        .onDisappear {
            animate = false
            pulseScale = 1.0
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("AI is thinking and processing the image")
    }
} 
