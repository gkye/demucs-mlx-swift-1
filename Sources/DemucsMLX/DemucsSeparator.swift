import Foundation
import MLX

public final class DemucsSeparator: @unchecked Sendable {
    public let modelName: String
    public let descriptor: DemucsModelDescriptor

    public var parameters: DemucsSeparationParameters

    private let model: StemSeparationModel

    public init(
        modelName: String = "htdemucs",
        parameters: DemucsSeparationParameters = DemucsSeparationParameters(),
        modelDirectory: URL? = nil
    ) throws {
        self.modelName = modelName
        self.descriptor = try DemucsModelRegistry.descriptor(for: modelName)
        self.parameters = try parameters.validated()
        self.model = try DemucsModelFactory.makeModel(for: descriptor, modelDirectory: modelDirectory)
    }

    public var sampleRate: Int {
        descriptor.sampleRate
    }

    public var audioChannels: Int {
        descriptor.audioChannels
    }

    public var sources: [String] {
        descriptor.sourceNames
    }

    public func updateParameters(_ parameters: DemucsSeparationParameters) throws {
        self.parameters = try parameters.validated()
    }

    public func separate(fileAt url: URL) throws -> DemucsSeparationResult {
        let audio = try AudioIO.loadAudio(from: url)
        return try separate(audio: audio)
    }

    public func separate(audio: DemucsAudio) throws -> DemucsSeparationResult {
        return try self.separate(audio: audio, monitor: nil)
    }

    // MARK: - Closure-Based Async API

    /// The internal serial queue used for background separation work.
    private static let separationQueue = DispatchQueue(label: "com.demucs.separation", qos: .userInitiated)

    /// Separate an audio file into stems on a background queue.
    ///
    /// - Parameters:
    ///   - url: Path to the audio file to separate.
    ///   - cancelToken: Optional token to request cancellation. Call `cancel()` on it to stop.
    ///   - interpolateProgress: If `true` (default), smoothly interpolates progress during GPU batch
    ///     execution gaps. If `false`, only reports raw progress updates from model sub-steps.
    ///   - progress: Optional progress callback. Called on the **main queue**.
    ///   - completion: Called on the **main queue** with the result or error.
    public func separate(
        fileAt url: URL,
        cancelToken: DemucsCancelToken?,
        interpolateProgress: Bool = true,
        progress: (@Sendable (_ progress: DemucsSeparationProgress) -> Void)?,
        completion: @escaping @Sendable (_ result: Result<DemucsSeparationResult, Error>) -> Void
    ) {
        let progressCopy = progress
        let completionCopy = completion
        DemucsSeparator.separationQueue.async(execute: { [self] in
            let result: Result<DemucsSeparationResult, Error>

            // Create interpolator for smooth progress during GPU batch gaps
            let interpolator: ProgressInterpolator?
            if interpolateProgress, let progressCopy {
                interpolator = ProgressInterpolator(callback: progressCopy)
            } else {
                interpolator = nil
            }

            do {
                let audio = try AudioIO.loadAudio(from: url)
                let monitor = SeparationMonitor(
                    cancelToken: cancelToken,
                    progressHandler: interpolator != nil
                        ? { @Sendable fraction, stage in interpolator?.onProgress(fraction, stage: stage) }
                        : Self.makeDirectProgressHandler(progressCopy)
                )
                let separationResult = try self.separate(audio: audio, monitor: monitor)
                result = .success(separationResult)
            }
            catch {
                result = .failure(error)
            }
            interpolator?.stop()
            DispatchQueue.main.async(execute: {
                completionCopy(result)
            })
        })
    }

    /// Separate audio into stems on a background queue.
    ///
    /// - Parameters:
    ///   - audio: The audio data to separate.
    ///   - cancelToken: Optional token to request cancellation.
    ///   - interpolateProgress: If `true` (default), smoothly interpolates progress during GPU batch
    ///     execution gaps. If `false`, only reports raw progress updates from model sub-steps.
    ///   - progress: Optional progress callback. Called on the **main queue**.
    ///   - completion: Called on the **main queue** with the result or error.
    public func separate(
        audio: DemucsAudio,
        cancelToken: DemucsCancelToken?,
        interpolateProgress: Bool = true,
        progress: (@Sendable (_ progress: DemucsSeparationProgress) -> Void)?,
        completion: @escaping @Sendable (_ result: Result<DemucsSeparationResult, Error>) -> Void
    ) {
        let progressCopy = progress
        let completionCopy = completion
        DemucsSeparator.separationQueue.async(execute: { [self] in
            let result: Result<DemucsSeparationResult, Error>

            let interpolator: ProgressInterpolator?
            if interpolateProgress, let progressCopy {
                interpolator = ProgressInterpolator(callback: progressCopy)
            } else {
                interpolator = nil
            }

            do {
                let monitor = SeparationMonitor(
                    cancelToken: cancelToken,
                    progressHandler: interpolator != nil
                        ? { @Sendable fraction, stage in interpolator?.onProgress(fraction, stage: stage) }
                        : Self.makeDirectProgressHandler(progressCopy)
                )
                let separationResult = try self.separate(audio: audio, monitor: monitor)
                result = .success(separationResult)
            }
            catch {
                result = .failure(error)
            }
            interpolator?.stop()
            DispatchQueue.main.async(execute: {
                completionCopy(result)
            })
        })
    }

    // MARK: - Progress Helpers

    /// Creates a direct (non-interpolated) progress handler that dispatches raw progress to the main queue.
    private static func makeDirectProgressHandler(
        _ callback: (@Sendable (DemucsSeparationProgress) -> Void)?
    ) -> (@Sendable (Float, String) -> Void)? {
        guard let callback else { return nil }
        let startTime = CFAbsoluteTimeGetCurrent()
        return { fraction, stage in
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let eta: TimeInterval? = fraction > 0.02 ? elapsed / Double(fraction) * Double(1 - fraction) : nil
            let progress = DemucsSeparationProgress(
                fraction: fraction,
                stage: stage,
                elapsedTime: elapsed,
                estimatedTimeRemaining: eta
            )
            DispatchQueue.main.async {
                callback(progress)
            }
        }
    }

    // MARK: - Internal

    private func separate(audio: DemucsAudio, monitor: SeparationMonitor?) throws -> DemucsSeparationResult {
        // Safety net: any synchronous MLX C++ error inside this block is converted
        // to a Swift throw. Without this, MLX's default behaviour is to call
        // `fatalError`, which crashes the app instead of letting the caller
        // present an error to the user.
        try MLX.withError {
            try _separate(audio: audio, monitor: monitor)
        }
    }

    private func _separate(audio: DemucsAudio, monitor: SeparationMonitor?) throws -> DemucsSeparationResult {
        let validated = try parameters.validated()

        try monitor?.checkCancellation()

        let input = audio.channelMajorSamples
        let remixed = AudioDSP.remixChannels(
            channelMajor: input,
            inputChannels: audio.channels,
            targetChannels: descriptor.audioChannels,
            frames: audio.frameCount
        )

        let resampled = AudioDSP.resampleChannelMajor(
            remixed,
            channels: descriptor.audioChannels,
            inputSampleRate: audio.sampleRate,
            targetSampleRate: descriptor.sampleRate,
            frames: audio.frameCount
        )

        let normalizedAudio = try DemucsAudio(
            channelMajor: resampled.samples,
            channels: descriptor.audioChannels,
            sampleRate: descriptor.sampleRate
        )

        try monitor?.checkCancellation()
        monitor?.reportProgress(0.0, stage: "Starting separation")

        let engine = SeparationEngine(model: model, parameters: validated, monitor: monitor)
        let stemsFlat = try engine.separate(
            mix: resampled.samples,
            channels: descriptor.audioChannels,
            frames: resampled.frames,
            sampleRate: descriptor.sampleRate
        )

        try monitor?.checkCancellation()
        monitor?.reportProgress(1.0, stage: "Complete")

        // stemsFlat is laid out as [source][channel][frame], contiguous. Each source's slice
        // is already the channel-major layout DemucsAudio expects, so we bulk-copy the slice
        // instead of scalar-looping 72M+ samples (which dominates the post-inference "Complete"
        // stage in Debug builds).
        let perSourceCount = descriptor.audioChannels * resampled.frames
        var stems: [String: DemucsAudio] = [:]
        for (sourceIndex, sourceName) in descriptor.sourceNames.enumerated() {
            let sourceStart = sourceIndex * perSourceCount
            let sourceSamples = Array(stemsFlat[sourceStart..<(sourceStart + perSourceCount)])

            stems[sourceName] = try DemucsAudio(
                channelMajor: sourceSamples,
                channels: descriptor.audioChannels,
                sampleRate: descriptor.sampleRate
            )
        }

        return DemucsSeparationResult(input: normalizedAudio, stems: stems)
    }
}
