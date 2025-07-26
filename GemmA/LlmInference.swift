//
//  LlmInference.swift
//  
//

import MediaPipeTasksGenAI
import Foundation
import ZIPFoundation
import CoreGraphics

/// Represents the single LLM model used in the app.
public enum ModelIdentifier: String, Identifiable {
    case gemma2B = "gemma-3n-E2B-it-int4"

    public var id: String { self.rawValue }
    public var fileName: String { "\(self.rawValue).task" }
    
    public var displayName: String {
        return "Gemma 3N (2B)"
    }

    /// Checks if the model is present in the app's main bundle.
    /// - Returns: True if the model file exists in the bundle.
    public static func isAvailableInBundle() -> Bool {
        return Bundle.main.path(forResource: ModelIdentifier.gemma2B.rawValue, ofType: "task") != nil
    }
}

/// Manages the on-device LLM, including its initialization, model file handling, and vision component extraction.
struct OnDeviceModel {
    private(set) var inference: LlmInference
    let identifier: ModelIdentifier = .gemma2B
    
    init() throws {
        let fileManager = FileManager.default
        let cacheDir = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        try fileManager.createDirectory(at: cacheDir, withIntermediateDirectories: true, attributes: nil)

        guard let bundleModelPath = Bundle.main.path(forResource: identifier.rawValue, ofType: "task") else {
            let errorMessage = "Critical Error: Model file '\(identifier.fileName)' not found in the app bundle. Please ensure it's added to the project and target."
            throw NSError(domain: "ModelSetupError", code: 1001, userInfo: [NSLocalizedDescriptionKey: errorMessage])
        }
        let modelCopyPath = cacheDir.appendingPathComponent(identifier.fileName)

        if !FileManager.default.fileExists(atPath: modelCopyPath.path) {
            try FileManager.default.copyItem(atPath: bundleModelPath, toPath: modelCopyPath.path)
        }

        let visionEncoderFileName = "TF_LITE_VISION_ENCODER"
        let visionAdapterFileName = "TF_LITE_VISION_ADAPTER"

        let extractedVisionEncoderPath = cacheDir.appendingPathComponent(visionEncoderFileName)
        let extractedVisionAdapterPath = cacheDir.appendingPathComponent(visionAdapterFileName)

        if !fileManager.fileExists(atPath: extractedVisionEncoderPath.path) ||
           !fileManager.fileExists(atPath: extractedVisionAdapterPath.path) {
            do {
                try OnDeviceModel.extractVisionModels(
                    fromArchive: modelCopyPath,
                    toDirectory: cacheDir,
                    filesToExtract: [visionEncoderFileName, visionAdapterFileName]
                )
            } catch {
                // Handle extraction error silently
            }
        }

        let options = LlmInference.Options(modelPath: modelCopyPath.path)
        options.maxTokens = 2048
        options.visionEncoderPath = extractedVisionEncoderPath.path
        options.visionAdapterPath = extractedVisionAdapterPath.path
        options.maxImages = 1

        inference = try LlmInference(options: options)
    }

    private static func extractVisionModels(fromArchive archiveURL: URL, toDirectory destinationURL: URL, filesToExtract: [String]) throws {
        let fileManager = FileManager.default
        let archive = try Archive(url: archiveURL, accessMode: .read)

        for fileName in filesToExtract {
            guard let entry = archive[fileName] else {
                continue
            }
            
            let destinationFilePath = destinationURL.appendingPathComponent(fileName)
            
            if fileManager.fileExists(atPath: destinationFilePath.path) {
                try fileManager.removeItem(at: destinationFilePath)
            }

            _ = try archive.extract(entry, to: destinationFilePath)
        }
    }
}

/// Represents a chat session with the loaded on-device LLM.
final class Chat {
    private let model: OnDeviceModel
    private var session: LlmInference.Session
    private let topK: Int
    private let topP: Float
    private let temperature: Float
    private let randomSeed: Int
    private let enableVisionModality: Bool
    private var needsSessionRecreation = false
    private var lastUsedParameters: (topK: Int, topP: Float, temperature: Float, randomSeed: Int)?
    
    // Token Management
    private let maxTokens: Int = 2048
    private let tokensPerInference: Int = 400
    private let maxInferencesBeforeReset: Int = 4
    private var inferenceCount: Int = 0
    private var isActivelyProcessing: Bool = false
    
    enum InferenceMode {
        case passive, active
    }
    
    init(model: OnDeviceModel, topK: Int = 20, topP: Float = 0.8, temperature: Float = 0.5, randomSeed: Int = 101, enableVisionModality: Bool = true) throws {
        self.model = model
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.randomSeed = randomSeed
        self.enableVisionModality = enableVisionModality
        let options = LlmInference.Session.Options()
        options.topk = topK
        options.topp = topP
        options.temperature = temperature
        options.enableVisionModality = enableVisionModality
        options.randomSeed = randomSeed
        session = try LlmInference.Session(llmInference: model.inference, options: options)
        inferenceCount = 0
    }
    
    public func resetContext() {
        guard !isActivelyProcessing else {
            return
        }
        
        let options = LlmInference.Session.Options()
        options.topk = topK
        options.topp = topP
        options.temperature = temperature
        options.enableVisionModality = enableVisionModality
        options.randomSeed = randomSeed
        session = try! LlmInference.Session(llmInference: model.inference, options: options)
        inferenceCount = 0
        needsSessionRecreation = false
        lastUsedParameters = (topK: topK, topP: topP, temperature: temperature, randomSeed: randomSeed)
        
        Thread.sleep(forTimeInterval: 0.2)
    }
    
    public func markForSessionRecreation() {
        needsSessionRecreation = true
    }
    
    private func checkInferenceCountAndResetIfNeeded() {
        guard !isActivelyProcessing else {
            return
        }
        
        if inferenceCount >= maxInferencesBeforeReset {
            resetContext()
        }
    }
    
    private func incrementInferenceCount() {
        inferenceCount += 1
    }
    
    func sendMessageSync(_ text: String, mode: InferenceMode = .passive) throws -> String {
        isActivelyProcessing = true
        defer { isActivelyProcessing = false }
        
        do {
            try session.addQueryChunk(inputText: text)
            let response = try session.generateResponse()
            incrementInferenceCount()
            return response
        } catch {
            resetContext()
            try session.addQueryChunk(inputText: text)
            let response = try session.generateResponse()
            incrementInferenceCount()
            return response
        }
    }

    public func addImageToQuery(image: CGImage) throws {
        if !isActivelyProcessing {
            checkInferenceCountAndResetIfNeeded()
        }
        try self.session.addImage(image: image)
    }
    
    func sendMessage(_ text: String, mode: InferenceMode = .passive) async throws -> AsyncThrowingStream<String, any Error> {
        isActivelyProcessing = true
        
        do {
            try session.addQueryChunk(inputText: text)
            let resultStream = session.generateResponseAsync()
            
            return AsyncThrowingStream { continuation in
                Task {
                    var accumulatedResponse = ""
                    do {
                        for try await chunk in resultStream {
                            accumulatedResponse += chunk
                            continuation.yield(chunk)
                        }
                        continuation.finish()
                        
                        self.incrementInferenceCount()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                    self.isActivelyProcessing = false
                }
            }
        } catch {
            if error.localizedDescription.contains("Roll back steps") || 
               error.localizedDescription.contains("current_step") {
                // Handle MediaPipe internal errors
            }
            resetContext()
            try session.addQueryChunk(inputText: text)
            let resultStream = session.generateResponseAsync()
            
            return AsyncThrowingStream { continuation in
                Task {
                    var accumulatedResponse = ""
                    do {
                        for try await chunk in resultStream {
                            accumulatedResponse += chunk
                            continuation.yield(chunk)
                        }
                        continuation.finish()
                        
                        self.incrementInferenceCount()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                    self.isActivelyProcessing = false
                }
            }
        }
    }

    public func getLastResponseGenerationTime() -> TimeInterval? {
        return self.session.metrics.responseGenerationTimeInSeconds
    }

    public func sizeInTokens(text: String) throws -> Int {
        return try self.session.sizeInTokens(text: text)
    }
}



