import Foundation
import UIKit



class VisionProcessor: ObservableObject {
    private var chat: Chat?
    private var currentSessionId: UUID?
    private var isProcessing: Bool = false
    
    private let combinedPrompt = """
    You are an AI visual assistant for a blind user. Briefly describe the path ahead and any obstacle blocking forward movement. If there are no obstructions, say "CLEAR." Use simple, everyday words. Respond in 7 words or fewer.
    some examples: curb, stairs, wall, person, pole, footpath, no obstacles, red DO NOT CROSS signal, green signal or safe crosswalk
    """
    
    init(chat: Chat) {
        self.chat = chat
        self.currentSessionId = UUID()
    }
    
    // MARK: - Input Validation
    private func validateInputs(images: [UIImage], prompt: String) -> (isValid: Bool, error: String?) {
        guard !images.isEmpty else {
            return (false, "No images provided")
        }
        
        // Only process the first image to prevent context corruption
        let image = images.first!
        
        guard let cgImage = image.cgImage else {
            return (false, "Image has invalid CGImage")
        }
        
        let size = CGSize(width: cgImage.width, height: cgImage.height)
        if size.width < 50 || size.height < 50 {
            return (false, "Image too small: \(Int(size.width))x\(Int(size.height))")
        }
        
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            return (false, "Prompt is empty")
        }
        
        guard trimmedPrompt.count <= 1000 else {
            return (false, "Prompt too long: \(trimmedPrompt.count) characters")
        }
        
        return (true, nil)
    }
    
    // MARK: - Session Management
    private func resetChatSession() {
        do {
            chat?.resetContext()
            currentSessionId = UUID()
        } catch {
            // Handle error silently
        }
    }
    
    // MARK: - Streaming inference for a batch of frames with custom prompt
    func processFramesStreaming(_ images: [UIImage], prompt: String, completion: @escaping (String, Bool) -> Void) {
        guard !isProcessing else {
            print("⚠️ VisionProcessor: Already processing, ignoring request")
            completion("Error: Already processing", true)
            return
        }
        
        isProcessing = true
        print("🔍 VisionProcessor: Starting processing, images count: \(images.count)")
        
        let validation = validateInputs(images: images, prompt: prompt)
        if !validation.isValid {
            isProcessing = false
            completion("Error: \(validation.error ?? "Invalid input")", true)
            return
        }
        
        let isActiveMode = prompt.contains("Answer the user's question about this image")
        let mode: Chat.InferenceMode = isActiveMode ? .active : .passive
        
        processWithRetry(images: images, prompt: prompt, mode: mode, completion: completion, retryCount: 0)
    }
    
    private func processWithRetry(images: [UIImage], prompt: String, mode: Chat.InferenceMode, completion: @escaping (String, Bool) -> Void, retryCount: Int) {
        let maxRetries = 1
        
        Task {
            let outputStream = await analyzeFramesStreaming(images, prompt: prompt, mode: mode)
            var response = ""
            var hasError = false
            var hasReceivedAnyOutput = false
            
            do {
                for try await chunk in outputStream {
                    response += chunk
                    
                    // Track if we've received any meaningful output
                    let trimmedChunk = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmedChunk.isEmpty {
                        hasReceivedAnyOutput = true
                    }
                    
                    completion(response, false)
                }
                
                if !hasError {
                    // Check if we got no meaningful output
                    if !hasReceivedAnyOutput || response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        completion("No response generated", true)
                    } else {
                        completion(response, true)
                    }
                }
            } catch {
                hasError = true
                
                if (error.localizedDescription.contains("OUT_OF_RANGE") || 
                    error.localizedDescription.contains("exceed context window") ||
                    error.localizedDescription.contains("current_step") ||
                    error.localizedDescription.contains("Roll back steps")) && retryCount < maxRetries {
                    
                    print("🔄 VisionProcessor: Retrying after error: \(error.localizedDescription)")
                    chat?.markForSessionRecreation()
                    chat?.resetContext()
                    
                    processWithRetry(images: images, prompt: prompt, mode: mode, completion: completion, retryCount: retryCount + 1)
                    return
                } else {
                    print("❌ VisionProcessor: Final error, not retrying: \(error.localizedDescription)")
                    completion("Processing error occurred: \(error.localizedDescription)", true)
                }
            }
            
            print("✅ VisionProcessor: Processing completed, resetting flag")
            isProcessing = false
        }
    }
    
    // Existing method for default prompt - always passive mode
    func processFramesStreaming(_ images: [UIImage], completion: @escaping (String, Bool) -> Void) {
        processFramesStreaming(images, prompt: combinedPrompt, completion: completion)
    }
    
    // MARK: - Core Analysis
    private func analyzeFramesStreaming(_ images: [UIImage], prompt: String, mode: Chat.InferenceMode) async -> AsyncThrowingStream<String, Error> {
        guard let chat = chat else {
            return AsyncThrowingStream { continuation in
                continuation.yield("Error: Model not ready")
                continuation.finish()
            }
        }
        
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Ensure we only process exactly one image
                    guard images.count == 1 else {
                        print("❌ VisionProcessor: Expected 1 image, got \(images.count)")
                        throw VisionProcessorError.invalidInput
                    }
                    
                    let image = images.first!
                    let uprightImage = image.fixedOrientation()
                    let squareImage = uprightImage.centerCroppedToSquare()
                    
                    print("🔍 VisionProcessor: Processing image, size: \(squareImage.size)")
                    print("🔍 VisionProcessor: Using prompt: \(prompt)")
                    
                    if let cgImage = squareImage.cgImage {
                        // Add image to query (context reset is handled by Chat class)
                        try chat.addImageToQuery(image: cgImage)
                        print("✅ VisionProcessor: Image added to query successfully")
                    } else {
                        throw VisionProcessorError.invalidImageData
                    }
                    
                    let stream = try await chat.sendMessage(prompt, mode: mode)
                    
                    for try await chunk in stream {
                        continuation.yield(chunk)
                    }
                    
                    continuation.finish()
                    
                } catch {
                    print("❌ VisionProcessor: Error during processing: \(error.localizedDescription)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Error Handling
    enum VisionProcessorError: Error, LocalizedError {
        case invalidImageData
        case modelNotReady
        case processingInProgress
        case invalidInput
        
        var errorDescription: String? {
            switch self {
            case .invalidImageData:
                return "Invalid image data"
            case .modelNotReady:
                return "Model not ready"
            case .processingInProgress:
                return "Processing already in progress"
            case .invalidInput:
                return "Invalid input parameters"
            }
        }
    }
}

// MARK: - UIImage Extensions
extension UIImage {
    func centerCroppedToSquare() -> UIImage {
        let originalSize = min(size.width, size.height)
        let x = (size.width - originalSize) / 2.0
        let y = (size.height - originalSize) / 2.0
        let cropRect = CGRect(x: x, y: y, width: originalSize, height: originalSize)
        guard let cgImage = self.cgImage?.cropping(to: cropRect) else { return self }
        return UIImage(cgImage: cgImage, scale: self.scale, orientation: self.imageOrientation)
    }
} 
