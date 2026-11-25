import CryptoKit
import Foundation
import MLX
import Metal
import os
import SwiftUI
import Tokenizers

/// The imported MLX graph and its evaluation must stay on one OS thread.
final class InklingMLXWorker: @unchecked Sendable {
    static let shared = InklingMLXWorker()
    private let condition = NSCondition()
    private var jobs: [() -> Void] = []

    private init() {
        let thread = Thread { [self] in
            while true {
                condition.lock()
                while jobs.isEmpty { condition.wait() }
                let job = jobs.removeFirst()
                condition.unlock()
                autoreleasepool(invoking: job)
            }
        }
        thread.name = "Inkling MLX"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func run<T: Sendable>(_ operation: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            condition.lock()
            jobs.append { continuation.resume(with: Result(catching: operation)) }
            condition.signal()
            condition.unlock()
        }
    }

    static func greedyToken(graph: URL, inputs: [[Int32]], profile: InklingTokenProfile? = nil) throws -> Int32 {
        let token = try prepareToken(graph: graph, inputs: inputs, profile: profile)
        if let profile {
            return try profile.measure("evaluate") { try withError { token.item(Int32.self) } }
        }
        return try withError { token.item(Int32.self) }
    }

    @inline(never)
    private static func prepareToken(graph: URL, inputs: [[Int32]], profile: InklingTokenProfile?) throws -> MLXArray {
        if let profile {
            let forward = try profile.measure("import") { try importFunction(from: graph) }
            return try profile.measure("invoke") { try invoke(inputs, using: forward) }
        }
        let forward = try importFunction(from: graph)
        return try invoke(inputs, using: forward)
    }

    private static func invoke(_ inputs: [[Int32]], using forward: ImportedFunction) throws -> MLXArray {
        let arrays = inputs.map { MLXArray($0) }
        let outputs = switch arrays.count {
        case 1: try forward(arrays[0])
        default: try forward(arrays[0], arrays[1])
        }
        guard let token = outputs.first, token.size == 1 else { throw EmbeddingProbe.Failure.invalidFixture }
        return token
    }
}

enum InklingChatMode {
    case standard, shortContext, extendedContext, recording

    var title: String {
        switch self {
        case .standard: "Inkling 2 Mini"
        case .shortContext: "Short-Context Chat"
        case .extendedContext: "1K-Context Chat"
        case .recording: "Recording Mode"
        }
    }

    func fixture(in directory: URL) -> URL {
        let name = switch self {
        case .standard, .recording: "inkling-full-decode"
        case .shortContext: "inkling-full-decode-64"
        case .extendedContext: "inkling-full-decode-1024"
        }
        return directory.appendingPathComponent(name, isDirectory: true)
    }
}

struct InklingChatRuntime {
    struct Configuration: Decodable {
        let version: Int
        let capacity: Int
        let padID: Int32
        let samplePrompt: String
        let sampleInputIDs: [Int]
        let sampleNextTokenID: Int32
    }

    let graph: URL
    let tokenizer: any Tokenizer
    let config: Configuration
    let stopIDs: Set<Int>
    let layerSession: InklingTokenProfile.ChatLayerSession?

    static func load(from directory: URL, preferLargerGraph: Bool = true, useLayerDecoder: Bool = false,
                     mode: InklingChatMode = .standard) async throws -> Self {
        let larger = directory.appendingPathComponent("inkling-chat-96.json")
        let configuration = preferLargerGraph && FileManager.default.fileExists(atPath: larger.path)
            ? larger : directory.appendingPathComponent("inkling-chat.json")
        let layerSession = try useLayerDecoder
            ? InklingTokenProfile.ChatLayerSession(directory: directory, mode: mode) : nil
        let config = try layerSession?.configuration
            ?? JSONDecoder().decode(Configuration.self, from: Data(contentsOf: configuration))
        guard config.version == 1, (layerSession != nil || [48, 64, 96, 256].contains(config.capacity)) else {
            throw EmbeddingProbe.Failure.invalidFixture
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory, strict: false)
        guard tokenizer.encode(text: config.samplePrompt, addSpecialTokens: false) == config.sampleInputIDs else {
            throw NSError(domain: "InklingChat", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "The local tokenizer did not match the checkpoint reference."
            ])
        }
        let stopIDs = Set([
            "<|endoftext|>", "<|end_message|>", "<|content_model_end_sampling|>",
            "<|message_model|>", "<|message_user|>", "<|message_system|>", "<|content_thinking|>",
        ].compactMap { tokenizer.convertTokenToId($0) })
        return Self(graph: configuration.deletingPathExtension().appendingPathExtension("mlxfn"), tokenizer: tokenizer,
                    config: config, stopIDs: stopIDs, layerSession: layerSession)
    }

    func prepare(_ chat: Conversation, message: String) throws -> ConversationContext.Prepared {
        try ConversationContext.prepare(chat, nextUserMessage: message, contextLimit: config.capacity,
                                        replyReserve: layerSession?.replyReserve ?? config.capacity / 3) {
            tokenizer.encode(text: $0, addSpecialTokens: false)
        }
    }

    func warmUp() async throws {
        guard let layerSession,
              let userContent = config.samplePrompt.range(of: "<|message_user|><|content_text|>") else { return }
        let prefix = String(config.samplePrompt[..<userContent.upperBound])
        let ids = tokenizer.encode(text: prefix, addSpecialTokens: false)
        guard !ids.isEmpty, config.sampleInputIDs.starts(with: ids) else {
            throw EmbeddingProbe.Failure.invalidFixture
        }
        _ = try await layerSession.nextToken(after: ids)
    }

    func nextToken(after ids: [Int]) async throws -> Int {
        guard !ids.isEmpty, ids.count <= config.capacity else { throw EmbeddingProbe.Failure.invalidFixture }
        if let layerSession { return try await layerSession.nextToken(after: ids) }
        let graph = graph
        let packed = ids.map(Int32.init) + Array(repeating: config.padID, count: config.capacity - ids.count)
        let position = [Int32(ids.count - 1)]
        return Int(try await InklingMLXWorker.shared.run {
            Memory.cacheLimit = 64 * 1024 * 1024
            return try InklingMLXWorker.greedyToken(graph: graph, inputs: [packed, position])
        })
    }
}

/// Shared layer decoding and launch benchmarks on the MLX worker thread.
final class InklingTokenProfile {
    private static let signposter = OSSignposter(subsystem: "local.inkling.probe", category: "Inference")
    private(set) var seconds: [String: Double] = [:]

    static func now() -> Double { ProcessInfo.processInfo.systemUptime }

    func measure<T>(_ name: StaticString, _ operation: () throws -> T) rethrows -> T {
        let interval = Self.signposter.beginInterval(name)
        let start = Self.now()
        defer {
            seconds[String(describing: name)] = Self.now() - start
            Self.signposter.endInterval(name, interval)
        }
        return try operation()
    }

    static func counters() -> [String: Any] {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        let process = ProcessInfo.processInfo
        var counters: [String: Any] = [
            "mlx_active_bytes": Memory.activeMemory, "mlx_cache_bytes": Memory.cacheMemory,
            "mlx_peak_bytes": Memory.peakMemory, "peak_resident_bytes": usage.ru_maxrss,
            "minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt,
            "block_inputs": usage.ru_inblock,
            "cpu_seconds": [usage.ru_utime, usage.ru_stime].reduce(0.0) {
                $0 + Double($1.tv_sec) + Double($1.tv_usec) / 1_000_000
            },
            "thermal_state": process.thermalState.rawValue, "low_power": process.isLowPowerModeEnabled,
        ]
        if result == KERN_SUCCESS {
            counters["physical_footprint_bytes"] = info.phys_footprint
            counters["compressed_bytes"] = info.compressed
        }
        if let device = MTLCreateSystemDefaultDevice() {
            counters["metal_allocated_bytes"] = device.currentAllocatedSize
        }
        return counters
    }

    struct Suite: Decodable {
        struct Prompt: Decodable {
            let name: String
            let text: String
            let ids: [Int]
        }
        let graphSHA256: String
        let configSHA256: String
        let tokenizerSHA256: String
        let checkpoint: String
        let source: String
        let optimization: String
        let capacity: Int
        let tokens: Int
        let repeats: Int
        let prompts: [Prompt]
    }

    struct CachedStep: Decodable {
        let inputToken: Int32
        let expectedToken: Int32
        let visits: Int

        enum CodingKeys: String, CodingKey {
            case inputToken = "input_token"
            case expectedToken = "expected_token"
            case visits
        }
    }

    static func cachedStep(directory: URL, retainGraph: Bool,
                           emit: @escaping @Sendable (String) -> Void) async throws {
        let fixture = directory.appendingPathComponent("inkling-cached-step")
        let reference = try JSONDecoder().decode(CachedStep.self, from: Data(contentsOf:
            fixture.appendingPathExtension("json")))
        try await InklingMLXWorker.shared.run {
            Memory.cacheLimit = 64 * 1024 * 1024
            let inputs = try loadArrays(url: fixture.appendingPathExtension("safetensors"))
            guard let token = inputs["token"], token.size == 1,
                  let kv = inputs["kv"], kv.shape.first == reference.visits,
                  let conv = inputs["conv"], try withError({ token.item(Int32.self) }) == reference.inputToken else {
                throw EmbeddingProbe.Failure.invalidFixture
            }
            let graph = fixture.appendingPathExtension("mlxfn")
            let setup = InklingTokenProfile()
            let retained = retainGraph ? try setup.measure("import") { try importFunction(from: graph) } : nil
            for iteration in 0..<(retainGraph ? 8 : 1) {
                let profile = InklingTokenProfile()
                let result = try autoreleasepool {
                    let outputs: [MLXArray]
                    if let retained {
                        outputs = try profile.measure("invoke") { try retained(token, kv, conv) }
                    } else {
                        outputs = try prepareCachedStep(graph: graph, token: token, kv: kv, conv: conv, profile: profile)
                    }
                    guard outputs.count == 3, outputs[0].size == 1,
                          outputs[1].shape == kv.shape, outputs[2].shape == conv.shape else {
                        throw EmbeddingProbe.Failure.invalidFixture
                    }
                    return try profile.measure("evaluate") {
                        try withError {
                            eval(outputs)
                            return outputs[0].item(Int32.self)
                        }
                    }
                }
                let data = try JSONSerialization.data(withJSONObject: [
                    "iteration": iteration, "retained_graph": retainGraph, "setup_phases": setup.seconds,
                    "id": result, "expected_id": reference.expectedToken,
                    "matched": result == reference.expectedToken, "phases": profile.seconds, "counters": counters(),
                ], options: .sortedKeys)
                emit("INKLING_CACHED_STEP " + String(decoding: data, as: UTF8.self))
                guard result == reference.expectedToken else { throw EmbeddingProbe.Failure.invalidFixture }
            }
        }
    }

    @inline(never)
    private static func prepareCachedStep(graph: URL, token: MLXArray, kv: MLXArray, conv: MLXArray,
                                          profile: InklingTokenProfile) throws -> [MLXArray] {
        let forward = try profile.measure("import") { try importFunction(from: graph) }
        return try profile.measure("invoke") { try forward(token, kv, conv) }
    }

    private struct LayerDecodeManifest: Decodable {
        struct Prompt: Decodable {
            let text: String
            let ids: [Int32]
            let expectedTokens: [Int32]
        }

        let checkpoint: String
        let prefixLength: Int?
        let inputToken: Int32?
        let expectedTokens: [Int32]?
        let prompts: [Prompt]?
        let visits: Int
        let boundaries: [Bool]
        let loopStart: Int
        let loopEnd: Int
        let loopPasses: Int
        let hiddenSize: Int
        let convolutionWidths: [Int]
        let kvCapacity: Int?
        let serialOnly: Bool?

        var capacity: Int { kvCapacity ?? 256 }

        var hasSupportedTopology: Bool {
            checkpoint == "54b4d8c2932ae5c0fc3b8a7a8c39f2633baef67e"
                && boundaries.count == 24 && hiddenSize == 2304
                && loopStart == 6 && loopEnd == 18 && loopPasses == 3
                && visits == 48 && convolutionWidths == [1024, 1024, 2304, 2304]
                && [64, 256, 1024].contains(capacity) && boundaries[loopStart]
        }
    }

    static let defaultAsyncLayerInterval = 8
    static let defaultLayerCacheMiB = 64

    enum LayerResidency: String {
        case none
        case recurrentSubset = "--resident"
        case recurrentSubsetAsync = "--resident-async"
        case all = "--all-resident"
        case allAsync = "--all-resident-async"
    }

    private final class LayerKernelPool {
        private let retainedNames: Set<String>
        private var retained: [URL: ImportedFunction] = [:]
        private(set) var imports = 0

        init(retainedLayers: Range<Int>) {
            retainedNames = Set(retainedLayers.map { String(format: "layer-%02d.mlxfn", $0) })
                .union(["embed.mlxfn", "head.mlxfn"])
        }

        func invoke(_ graph: URL, batch: LayerBatch?,
                    _ body: (ImportedFunction) throws -> [MLXArray]) throws -> [MLXArray] {
            if let loaded = retained[graph] { return try body(loaded) }
            imports += 1
            guard retainedNames.contains(graph.lastPathComponent) else {
                if let batch {
                    let loaded = try importFunction(from: graph)
                    batch.functions.append(loaded)
                    return try body(loaded)
                }
                return try invokeLayerKernel(graph, body)
            }
            let loaded = try importFunction(from: graph)
            retained[graph] = loaded
            return try body(loaded)
        }
    }

    private final class LayerBatch {
        var functions: [ImportedFunction] = []
        var outputs: [MLXArray] = []

        func wait() throws {
            try withExtendedLifetime(functions) {
                try withError { eval(outputs) }
            }
        }
    }

    private struct LayerDecodeCache {
        var states: [[MLXArray]]

        init(empty manifest: LayerDecodeManifest) {
            let initial = [MLXArray.zeros([2, 8, manifest.capacity, 128], dtype: .float16)]
                + manifest.convolutionWidths.map { MLXArray.zeros([1, 3, $0], dtype: .float32) }
            states = Array(repeating: initial, count: manifest.visits)
        }

        init(file: URL, manifest: LayerDecodeManifest) throws {
            let arrays = try loadArrays(url: file)
            states = try (0..<manifest.visits).map { slot in
                guard let kv = arrays["kv\(slot)"], kv.shape == [2, 8, manifest.capacity, 128], kv.dtype == .float16 else {
                    throw EmbeddingProbe.Failure.invalidFixture
                }
                let convolution = try manifest.convolutionWidths.enumerated().map { index, width in
                    guard let value = arrays["conv\(slot)_\(index)"],
                          value.shape == [1, 3, width], value.dtype == .float32 else {
                        throw EmbeddingProbe.Failure.invalidFixture
                    }
                    return value
                }
                return [kv] + convolution
            }
        }

        func errors(against reference: Self) throws -> (kv: [Float], convolution: [Float]) {
            try withError {
                var kvErrors: [Float] = []
                var convolutionErrors: [Float] = []
                for (actual, expected) in zip(states, reference.states) {
                    let errors = try zip(actual, expected).map { pair in
                        let error = (pair.0.asType(.float32) - pair.1.asType(.float32)).abs().max().item(Float.self)
                        guard error.isFinite else { throw EmbeddingProbe.Failure.invalidFixture }
                        return error
                    }
                    kvErrors.append(errors[0])
                    convolutionErrors.append(errors.dropFirst().max() ?? 0)
                }
                return (kvErrors, convolutionErrors)
            }
        }

        func digest() -> String {
            var hash = SHA256()
            for state in states {
                for tensor in state {
                    hash.update(data: tensor.asData(access: .copy).data)
                }
            }
            return hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
    }

    final class ChatLayerSession: @unchecked Sendable {
        let configuration: InklingChatRuntime.Configuration
        let replyReserve: Int
        private let residency: LayerResidency
        private let fixture: URL
        private let manifest: LayerDecodeManifest
        // The worker owns all mutable inference state.
        private var pool: LayerKernelPool?
        private var cache: LayerDecodeCache?
        private var cachedIDs: [Int] = []
        private var prediction: Int?

        init(directory: URL, mode: InklingChatMode) throws {
            let fixture = mode.fixture(in: directory)
            let manifest = try JSONDecoder().decode(LayerDecodeManifest.self, from: Data(contentsOf:
                fixture.appendingPathComponent("manifest.json")))
            guard manifest.hasSupportedTopology,
                  mode != .shortContext || manifest.capacity == 64,
                  mode != .extendedContext || manifest.capacity == 1024,
                  let sample = manifest.prompts?.first,
                  let expected = sample.expectedTokens.first else {
                throw EmbeddingProbe.Failure.invalidFixture
            }
            self.fixture = fixture
            self.manifest = manifest
            residency = mode == .recording ? .recurrentSubset : .allAsync
            replyReserve = switch mode {
            case .standard: manifest.capacity == 64 ? 16 : 64
            case .shortContext: 16
            case .extendedContext: 128
            case .recording: 8
            }
            configuration = .init(version: 1, capacity: mode == .recording ? 64 : manifest.capacity, padID: 0, samplePrompt: sample.text,
                                  sampleInputIDs: sample.ids.map(Int.init), sampleNextTokenID: expected)
        }

        func nextToken(after ids: [Int]) async throws -> Int {
            let prefix = try await InklingMLXWorker.shared.run {
                if !ids.starts(with: self.cachedIDs) || (ids.count == self.cachedIDs.count && self.prediction == nil) {
                    self.cache = nil
                    self.cachedIDs.removeAll()
                    self.prediction = nil
                }
                return self.cachedIDs.count
            }
            for position in prefix..<ids.count {
                try Task.checkCancellation()
                let id = ids[position]
                try await InklingMLXWorker.shared.run {
                    try self.consume(id, position: position, predict: position == ids.count - 1)
                }
            }
            return try await InklingMLXWorker.shared.run {
                guard let prediction = self.prediction else { throw EmbeddingProbe.Failure.invalidFixture }
                return prediction
            }
        }

        private func consume(_ id: Int, position: Int, predict: Bool) throws {
            if pool == nil {
                Memory.cacheLimit = defaultLayerCacheMiB * 1024 * 1024
                let retainedLayers = residency == .recurrentSubset
                    ? manifest.loopStart..<(manifest.loopStart + 6) : manifest.boundaries.indices
                pool = LayerKernelPool(retainedLayers: retainedLayers)
            }
            if cache == nil { cache = LayerDecodeCache(empty: manifest) }
            do {
                // Single-position replay supports arbitrary prompt lengths with a smaller prefill working set.
                let (next, _) = try decodeLayerStep(
                    tokens: MLXArray([Int32(id)]), position: MLXArray([Int32(position)]),
                    cache: &cache!, manifest: manifest, fixture: fixture, pool: pool,
                    asyncLayerInterval: residency == .allAsync ? defaultAsyncLayerInterval : nil,
                    predict: predict)
                prediction = try next.map { token in Int(try withError { token.item(Int32.self) }) }
                cachedIDs.append(id)
            } catch {
                cache = nil
                cachedIDs.removeAll()
                prediction = nil
                throw error
            }
        }
    }

    static func layerDecode(directory: URL, emit: @escaping @Sendable (String) -> Void) async throws {
        let fixture = directory.appendingPathComponent("inkling-layer-decode", isDirectory: true)
        let manifest = try JSONDecoder().decode(LayerDecodeManifest.self, from: Data(contentsOf:
            fixture.appendingPathComponent("manifest.json")))
        guard manifest.hasSupportedTopology,
              let expectedTokens = manifest.expectedTokens, (2...8).contains(expectedTokens.count),
              let inputToken = manifest.inputToken, let prefixLength = manifest.prefixLength,
              (1...248).contains(prefixLength) else {
            throw EmbeddingProbe.Failure.invalidFixture
        }
        try await InklingMLXWorker.shared.run {
            Memory.cacheLimit = 64 * 1024 * 1024
            var cache = try LayerDecodeCache(file: fixture.appendingPathComponent("initial.safetensors"),
                                             manifest: manifest)
            var token = MLXArray([inputToken])
            var elapsed = 0.0
            var firstStepSeconds = 0.0
            var allStepsMatched = true

            func record(_ values: [String: Any]) throws {
                let data = try JSONSerialization.data(withJSONObject: values, options: .sortedKeys)
                emit("INKLING_LAYER_DECODE " + String(decoding: data, as: UTF8.self))
            }

            for (step, expected) in expectedTokens.enumerated() {
                let start = now()
                let tag = MLXArray.zeros([step + 1], dtype: .int32)
                let (prediction, visited) = try decodeLayerStep(tokens: token, position: tag, cache: &cache,
                                                          manifest: manifest, fixture: fixture)
                let seconds = now() - start
                elapsed += seconds
                if step == 0 { firstStepSeconds = seconds }
                guard let next = prediction else { throw EmbeddingProbe.Failure.invalidFixture }
                let id = try withError { next.item(Int32.self) }
                let measurements = counters()
                let reference = try LayerDecodeCache(
                    file: fixture.appendingPathComponent("expected-\(step).safetensors"), manifest: manifest)
                let errors = try cache.errors(against: reference)
                let kvError = errors.kv.max() ?? 0
                let convolutionError = errors.convolution.max() ?? 0
                let matched = id == expected && kvError <= 0.002 && convolutionError <= 0.002
                allStepsMatched = allStepsMatched && matched
                try record([
                    "event": "step", "step": step, "id": id, "expected_id": expected,
                    "position": prefixLength + step, "visits": visited, "seconds": seconds,
                    "kv_max_abs_error": kvError, "convolution_max_abs_error": convolutionError,
                    "kv_error_by_slot": errors.kv, "convolution_error_by_slot": errors.convolution,
                    "matched": matched, "counters": measurements,
                ])
                token = next.reshaped(1)
            }
            try record(["event": "complete", "tokens": expectedTokens.count,
                        "parity_passed": allStepsMatched,
                        "tokens_per_second": Double(expectedTokens.count) / elapsed,
                        "post_first_tokens_per_second": Double(expectedTokens.count - 1)
                            / (elapsed - firstStepSeconds)])
        }
    }

    static func fullDecode(directory: URL, fixtureDirectory: URL? = nil, residency: LayerResidency,
                           fingerprintCache: Bool = false,
                           asyncLayerInterval: Int = defaultAsyncLayerInterval,
                           cacheMiB: Int = defaultLayerCacheMiB,
                           serialPrefill: Bool = false, predictEachPromptToken: Bool = false,
                           emit: @escaping @Sendable (String) -> Void) async throws {
        let fixture = fixtureDirectory ?? directory.appendingPathComponent("inkling-full-decode", isDirectory: true)
        let manifest = try JSONDecoder().decode(LayerDecodeManifest.self, from: Data(contentsOf:
            fixture.appendingPathComponent("manifest.json")))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory, strict: false)
        let serialPrefill = serialPrefill || manifest.serialOnly == true
        guard manifest.hasSupportedTopology, let prompts = manifest.prompts, !prompts.isEmpty,
              prompts.allSatisfy({ prompt in
                  prompt.ids.count > 1 && (2...64).contains(prompt.expectedTokens.count)
                      && prompt.ids.count + prompt.expectedTokens.count <= manifest.capacity
                      && tokenizer.encode(text: prompt.text, addSpecialTokens: false) == prompt.ids.map(Int.init)
              }) else { throw EmbeddingProbe.Failure.invalidFixture }

        try await InklingMLXWorker.shared.run {
            Memory.cacheLimit = cacheMiB * 1024 * 1024
            let retainedLayers: Range<Int>? = switch residency {
            case .none: nil
            case .recurrentSubset, .recurrentSubsetAsync: manifest.loopStart..<(manifest.loopStart + 6)
            case .all, .allAsync: manifest.boundaries.indices
            }
            let pool = retainedLayers.map { LayerKernelPool(retainedLayers: $0) }
            let submissionInterval = residency == .allAsync ? asyncLayerInterval : nil
            let boundedAsync = residency == .recurrentSubsetAsync
            func record(_ values: [String: Any]) throws {
                let data = try JSONSerialization.data(withJSONObject: values, options: .sortedKeys)
                emit("INKLING_FULL_DECODE " + String(decoding: data, as: UTF8.self))
            }
            try record(["event": "start", "prompt_count": prompts.count,
                        "retained_layers": retainedLayers.map(Array.init) ?? [],
                        "async_layer_interval": submissionInterval ?? (boundedAsync ? 2 : 0),
                        "fingerprint_cache": fingerprintCache, "cache_limit_mib": cacheMiB,
                        "kv_capacity": manifest.capacity, "serial_prefill": serialPrefill,
                        "predict_each_prompt_token": predictEachPromptToken, "counters": counters()])
            for (promptIndex, prompt) in prompts.enumerated() {
                var cache = LayerDecodeCache(empty: manifest)
                var token = MLXArray(prompt.ids)
                var firstMismatch: Int?
                var decodeSeconds = 0.0
                var firstTokenSeconds = 0.0
                for (index, expected) in prompt.expectedTokens.enumerated() {
                    let position = index == 0 ? (serialPrefill ? prompt.ids.count - 1 : 0)
                        : prompt.ids.count + index - 1
                    let previousImports = pool?.imports ?? 0
                    let start = now()
                    if index == 0 && serialPrefill {
                        for offset in 0..<position {
                            _ = try decodeLayerStep(
                                tokens: MLXArray([prompt.ids[offset]]), position: MLXArray([Int32(offset)]), cache: &cache,
                                manifest: manifest, fixture: fixture, pool: pool, asyncLayerInterval: submissionInterval,
                                boundedAsync: boundedAsync, predict: predictEachPromptToken)
                        }
                        token = MLXArray([prompt.ids[position]])
                    }
                    let (prediction, visits) = try decodeLayerStep(
                        tokens: token, position: MLXArray([Int32(position)]), cache: &cache,
                        manifest: manifest, fixture: fixture, pool: pool, asyncLayerInterval: submissionInterval,
                        boundedAsync: boundedAsync)
                    guard let next = prediction else { throw EmbeddingProbe.Failure.invalidFixture }
                    let id = try withError { next.item(Int32.self) }
                    let elapsed = now() - start
                    if index == 0 { firstTokenSeconds = elapsed } else { decodeSeconds += elapsed }
                    if id != expected && firstMismatch == nil { firstMismatch = index }
                    var event: [String: Any] = [
                        "event": "token", "prompt": promptIndex, "index": index, "id": id,
                        "expected_id": expected, "position": position, "visits": visits,
                        "seconds": elapsed, "counters": counters(),
                    ]
                    if let pool { event["graph_imports"] = pool.imports - previousImports }
                    if fingerprintCache { event["cache_sha256"] = cache.digest() }
                    try record(event)
                    token = next.reshaped(1)
                }
                try record(["event": "prompt_complete", "prompt": promptIndex, "input_tokens": prompt.ids.count,
                            "output_tokens": prompt.expectedTokens.count, "ttft_seconds": firstTokenSeconds,
                            "decode_tokens_per_second": Double(prompt.expectedTokens.count - 1) / decodeSeconds,
                            "first_mismatch": firstMismatch.map { $0 as Any } ?? NSNull(), "counters": counters()])
            }
            try record(["event": "complete"])
        }
    }

    private static func decodeLayerStep(
        tokens: MLXArray, position: MLXArray, cache: inout LayerDecodeCache,
        manifest: LayerDecodeManifest, fixture: URL, pool: LayerKernelPool? = nil,
        asyncLayerInterval: Int? = nil, boundedAsync: Bool = false, predict: Bool = true
    ) throws -> (MLXArray?, Int) {
        let asyncLayers = asyncLayerInterval != nil
        let length = tokens.size
        let emptyHidden = MLXArray.zeros([0, manifest.hiddenSize], dtype: .float32)
        let emptyBlocks = MLXArray.zeros([0, length, manifest.hiddenSize], dtype: .float32)
        var hidden = try runLayerKernel(fixture.appendingPathComponent("embed.mlxfn"), count: 1,
                                        pool: pool, synchronous: !asyncLayers) {
            try $0(tokens)
        }[0]
        var blocks = emptyBlocks
        var visited = Set<Int>()
        var batch = boundedAsync ? LayerBatch() : nil
        var inFlight: LayerBatch?

        func visit(_ layer: Int, slot: Int, priorHidden: MLXArray, priorBlocks: MLXArray) throws {
            guard hidden.shape == [length, manifest.hiddenSize], visited.insert(slot).inserted else {
                throw EmbeddingProbe.Failure.invalidFixture
            }
            let inputs = cache.states[slot]
            let graph = fixture.appendingPathComponent(String(format: "layer-%02d.mlxfn", layer))
            let output = try runLayerKernel(graph, count: inputs.count + 1,
                                            pool: pool, synchronous: !asyncLayers && !boundedAsync,
                                            batch: batch) { forward in
                try forward(position, hidden, blocks, priorHidden, priorBlocks,
                            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
            }
            guard zip(output.dropFirst(), inputs).allSatisfy({ $0.shape == $1.shape && $0.dtype == $1.dtype }) else {
                throw EmbeddingProbe.Failure.invalidFixture
            }
            if manifest.boundaries[layer] {
                let residual = hidden.expandedDimensions(axis: 0)
                blocks = blocks.shape[0] == 0 ? residual : concatenated([blocks, residual], axis: 0)
            }
            hidden = output[0]
            cache.states[slot] = Array(output.dropFirst())
            if let prepared = batch {
                prepared.outputs.append(contentsOf: output)
                if visited.count.isMultiple(of: 2) && visited.count < manifest.visits {
                    // Retain at most two visits in flight and two more being prepared.
                    try inFlight?.wait()
                    try withError { asyncEval(prepared.outputs) }
                    inFlight = prepared
                    batch = LayerBatch()
                }
            }
            if let asyncLayerInterval, visited.count.isMultiple(of: asyncLayerInterval),
               visited.count < manifest.visits {
                try withError { asyncEval([hidden]) }
            }
        }

        for layer in 0..<manifest.loopStart {
            try visit(layer, slot: layer, priorHidden: emptyHidden, priorBlocks: emptyBlocks)
        }
        let preLoopHidden = hidden
        let preLoopBlocks = blocks
        let loopLength = manifest.loopEnd - manifest.loopStart
        for pass in 0..<manifest.loopPasses {
            let priorHidden = pass == 0 ? emptyHidden : hidden
            let firstPriorBlock = preLoopBlocks.shape[0] + 1
            let priorBlocks = pass > 0 && firstPriorBlock < blocks.shape[0]
                ? blocks[firstPriorBlock...] : emptyBlocks
            hidden = preLoopHidden
            blocks = preLoopBlocks
            for offset in 0..<loopLength {
                let layer = manifest.loopStart + offset
                let slot = pass == 0 ? layer : manifest.boundaries.count + (pass - 1) * loopLength + offset
                try visit(layer, slot: slot, priorHidden: priorHidden, priorBlocks: priorBlocks)
            }
        }
        for layer in manifest.loopEnd..<manifest.boundaries.count {
            try visit(layer, slot: layer, priorHidden: emptyHidden, priorBlocks: emptyBlocks)
        }
        guard visited.count == manifest.visits else { throw EmbeddingProbe.Failure.invalidFixture }
        try inFlight?.wait()
        var next: MLXArray?
        if predict {
            let output = try runLayerKernel(fixture.appendingPathComponent("head.mlxfn"), count: 1,
                                            pool: pool, synchronous: !asyncLayers && !boundedAsync) {
                try $0(hidden, blocks)
            }[0]
            guard output.size == 1 else { throw EmbeddingProbe.Failure.invalidFixture }
            next = output.asType(.int32)
        }
        if asyncLayers || boundedAsync {
            // Commit the cache before advancing the prompt, even when there is no prediction.
            let outputs = (next.map { [$0] } ?? []) + cache.states.flatMap { $0 }
            try withExtendedLifetime(batch) {
                try withError { eval(outputs) }
            }
        }
        return (next, visited.count)
    }

    private static func runLayerKernel(
        _ graph: URL, count: Int, pool: LayerKernelPool?, synchronous: Bool, batch: LayerBatch? = nil,
        _ invoke: (ImportedFunction) throws -> [MLXArray]
    ) throws -> [MLXArray] {
        try autoreleasepool {
            let outputs = try pool.map { try $0.invoke(graph, batch: batch, invoke) } ?? invokeLayerKernel(graph, invoke)
            guard outputs.count == count else { throw EmbeddingProbe.Failure.invalidFixture }
            if synchronous { try withError { eval(outputs) } }
            return outputs
        }
    }

    @inline(never)
    private static func invokeLayerKernel(
        _ graph: URL, _ invoke: (ImportedFunction) throws -> [MLXArray]
    ) throws -> [MLXArray] {
        try invoke(importFunction(from: graph))
    }

    static func benchmark(directory: URL, emit: @escaping @Sendable (String) -> Void) async throws {
        let suite = try JSONDecoder().decode(Suite.self, from: Data(contentsOf:
            directory.appendingPathComponent("inkling-benchmark.json")))
        let start = now()
        let runtime = try await InklingChatRuntime.load(from: directory, preferLargerGraph: suite.capacity != 48)
        let setupSeconds = now() - start
        guard suite.capacity == runtime.config.capacity, suite.tokens > 1, suite.repeats > 0,
              !suite.prompts.isEmpty, suite.prompts.allSatisfy({
                  !$0.ids.isEmpty && $0.ids.count <= suite.capacity - suite.tokens &&
                  runtime.tokenizer.encode(text: $0.text, addSpecialTokens: false) == $0.ids
              }) else { throw EmbeddingProbe.Failure.invalidFixture }
        let config = runtime.config
        let graph = runtime.graph
        let sampleNextToken = Int(config.sampleNextTokenID)
        try await InklingMLXWorker.shared.run {
            func record(_ values: [String: Any]) throws {
                let data = try JSONSerialization.data(withJSONObject: values, options: .sortedKeys)
                emit("INKLING_BENCH " + String(decoding: data, as: UTF8.self))
            }
            Memory.cacheLimit = 64 * 1024 * 1024
            var model = utsname()
            uname(&model)
            let hardware = withUnsafeBytes(of: model.machine) {
                String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self)
            }
            try record([
                "event": "start", "backend": "graph", "capacity": suite.capacity,
                "graph_sha256": suite.graphSHA256, "config_sha256": suite.configSHA256,
                "tokenizer_sha256": suite.tokenizerSHA256, "checkpoint": suite.checkpoint,
                "source": suite.source, "optimization": suite.optimization, "optimized": !_isDebugAssertConfiguration(),
                "device": hardware, "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "kv": "no reuse", "recurrent_passes": 3, "experts_per_token": 8,
                "sampling": "greedy fixed count", "drafting": false, "cache_limit_bytes": Memory.cacheLimit,
                "setup_seconds": setupSeconds, "counters": counters(),
            ])
            for repetition in 0..<suite.repeats {
                for prompt in suite.prompts {
                    try record(["event": "prompt", "name": prompt.name, "repeat": repetition,
                                "input_ids": prompt.ids, "tokens": suite.tokens, "counters": counters()])
                    var ids = prompt.ids
                    let promptStart = now()
                    for step in 0..<suite.tokens {
                        try Task.checkCancellation()
                        let profile = InklingTokenProfile()
                        let packed = ids.map(Int32.init) + Array(repeating: config.padID, count: config.capacity - ids.count)
                        let tokenStart = now()
                        let token: Int
                        do {
                            token = Int(try autoreleasepool {
                                try InklingMLXWorker.greedyToken(
                                    graph: graph, inputs: [packed, [Int32(ids.count - 1)]], profile: profile)
                            })
                        } catch {
                            try record(["event": "failure", "step": step, "phases": profile.seconds,
                                        "counters": counters(), "error": error.localizedDescription])
                            throw error
                        }
                        let end = now()
                        if step == 0, prompt.ids == config.sampleInputIDs, token != sampleNextToken {
                            throw EmbeddingProbe.Failure.invalidFixture
                        }
                        ids.append(token)
                        try record(["event": "token", "step": step, "id": token,
                                    "seconds": end - tokenStart, "since_prompt_seconds": end - promptStart,
                                    "phases": profile.seconds, "counters": counters()])
                    }
                }
            }
            try record(["event": "complete"])
        }
    }
}

#if os(iOS)
struct InklingChatView: View {
    let directory: URL
    let mode: InklingChatMode

    @AppStorage("inklingChatID") private var savedChatID = ""
    @State private var conversation = Conversation()
    @State private var runtime: InklingChatRuntime?
    @State private var warmingUp = false
    @State private var draft = ""
    @State private var pending = ""
    @State private var reply = ""
    @State private var generation: Task<Void, Never>?
    @State private var error: String?
    @State private var omitted = 0
    @State private var showingSessions = false
    @State private var sessions: [Conversation] = []
    @State private var sessionError: String?

    var body: some View {
        ScrollViewReader { scroll in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(conversation.turns.indices, id: \.self) { index in
                        bubble(conversation.turns[index].user, user: true)
                        bubble(conversation.turns[index].assistant, user: false)
                        if let metrics = conversation.turns[index].metrics {
                            replyMetrics(metrics)
                        }
                    }
                    if !pending.isEmpty {
                        bubble(pending, user: true)
                        if reply.isEmpty { ProgressView("Generating reply") }
                        else { bubble(reply, user: false) }
                    } else if conversation.turns.isEmpty, runtime != nil {
                        ContentUnavailableView(
                            "Chat with Inkling 2 Mini", systemImage: "bubble.left",
                            description: Text(mode == .recording
                                ? "Replies are slower and limited to eight tokens."
                                : "Send a short message. Replies run on your iPhone."))
                    }
                    Color.clear.frame(height: 1).id("latest")
                }
                .padding()
            }
            .onChange(of: reply) { _, _ in scroll.scrollTo("latest", anchor: .bottom) }
            .onChange(of: pending) { _, _ in scroll.scrollTo("latest", anchor: .bottom) }
            .onChange(of: conversation.id) { _, _ in scroll.scrollTo("latest", anchor: .bottom) }
        }
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 8) {
                if let error { Text(error).font(.footnote).foregroundStyle(.red) }
                if runtime == nil && error == nil {
                    ProgressView(warmingUp ? "Warming up Inkling 2 Mini" : "Loading local tokenizer")
                }
                if omitted > 0 && !pending.isEmpty {
                    Text("Older turns omitted from this prompt: \(omitted)").font(.caption).foregroundStyle(.secondary)
                }
                HStack(alignment: .bottom) {
                    TextField("Message Inkling 2 Mini", text: $draft, axis: .vertical)
                        .lineLimit(1...4)
                        .textFieldStyle(.roundedBorder)
                        .disabled(generation != nil)
                    if let generation {
                        Button("Stop", systemImage: "stop.circle") { generation.cancel() }
                            .labelStyle(.iconOnly)
                    } else {
                        Button("Send", systemImage: "arrow.up.circle.fill", action: send)
                            .labelStyle(.iconOnly)
                            .disabled(runtime == nil || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
                .font(.title3)
            }
            .padding()
            .background(.regularMaterial)
        }
        .navigationTitle(mode.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button("Sessions", systemImage: "list.bullet") { openSessions() }
                    .disabled(generation != nil)
                Button("New chat", systemImage: "square.and.pencil") { newSession() }
                    .disabled(generation != nil)
            }
        }
        .sheet(isPresented: $showingSessions) {
            NavigationStack {
                List {
                    if let sessionError { Text(sessionError).foregroundStyle(.red) }
                    if sessions.isEmpty && sessionError == nil {
                        Text("Saved chats appear here after your first reply.").foregroundStyle(.secondary)
                    }
                    ForEach(sessions) { session in
                        Button {
                            switchSession(to: session)
                            showingSessions = false
                        } label: {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(session.turns.first?.user ?? "New chat").lineLimit(1)
                                    Text("\(session.turns.count) turns").font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if session.id == conversation.id { Image(systemName: "checkmark") }
                            }
                        }
                    }
                }
                .navigationTitle("Sessions")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Done") { showingSessions = false }
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Button("New chat") {
                            newSession()
                            showingSessions = false
                        }
                    }
                }
            }
        }
        .task {
            do {
                if let id = UUID(uuidString: savedChatID) { conversation = try ConversationStore.local().load(id) }
                let loaded = try await InklingChatRuntime.load(
                    from: directory, useLayerDecoder: true, mode: mode)
                warmingUp = true
                defer { warmingUp = false }
                try await loaded.warmUp()
                try Task.checkCancellation()
                runtime = loaded
            } catch is CancellationError {
            } catch { self.error = error.localizedDescription }
        }
        .onDisappear { generation?.cancel() }
    }

    private func openSessions() {
        sessionError = nil
        sessions = []
        do {
            let store = try ConversationStore.local()
            for id in try store.list() {
                do { sessions.append(try store.load(id)) }
                catch { sessionError = "A saved chat could not be opened." }
            }
        } catch { sessionError = error.localizedDescription }
        showingSessions = true
    }

    private func newSession() {
        switchSession(to: Conversation())
    }

    private func switchSession(to session: Conversation) {
        conversation = session
        savedChatID = session.turns.isEmpty ? "" : session.id.uuidString
        draft = ""
        error = nil
        omitted = 0
    }

    private func bubble(_ text: String, user: Bool) -> some View {
        Text(text)
            .padding(12)
            .background(user ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12),
                        in: RoundedRectangle(cornerRadius: 14))
            .frame(maxWidth: .infinity, alignment: user ? .trailing : .leading)
            .textSelection(.enabled)
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    private func replyMetrics(_ metrics: Conversation.Turn.Metrics) -> some View {
        let precision = FloatingPointFormatStyle<Double>.number.precision(.fractionLength(2))
        return VStack(alignment: .leading, spacing: 3) {
            Text("\(metrics.generatedTokens) reply tokens · \(metrics.tokensPerSecond.formatted(precision)) tok/s")
            Text("First text \(metrics.firstTextSeconds.formatted(precision)) s · Total \(metrics.elapsedSeconds.formatted(precision)) s")
            Text("\(metrics.promptTokens) prompt tokens")
            if metrics.reachedContextLimit { Text("Stopped at the context limit.") }
            if metrics.omittedTurns > 0 { Text("Older turns omitted: \(metrics.omittedTurns)") }
        }
        .font(.caption)
        .monospacedDigit()
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .combine)
    }

    private func send() {
        guard let runtime, generation == nil else { return }
        let message = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        error = nil
        do {
            let prompt = try runtime.prepare(conversation, message: message)
            omitted = prompt.omittedTurns
            draft = ""
            pending = message
            generation = Task {
                defer {
                    generation = nil
                    pending = ""
                    reply = ""
                }
                do {
                    let start = ContinuousClock.now
                    var firstText: ContinuousClock.Instant?
                    var tokens: [Int] = []
                    let available = runtime.config.capacity - prompt.tokens.count
                    let limit = mode == .recording ? min(available, prompt.reservedReplyTokens) : available
                    for _ in 0..<limit {
                        try Task.checkCancellation()
                        let token = try await runtime.nextToken(after: prompt.tokens + tokens)
                        try Task.checkCancellation()
                        if runtime.stopIDs.contains(token) { break }
                        tokens.append(token)
                        reply = runtime.tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                        if firstText == nil && !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            firstText = .now
                        }
                    }
                    let end = ContinuousClock.now
                    guard let firstText, !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        throw NSError(domain: "InklingChat", code: 2, userInfo: [
                            NSLocalizedDescriptionKey: "Inkling 2 Mini ended the reply before generating text."
                        ])
                    }
                    var updated = conversation
                    let metrics = Conversation.Turn.Metrics(
                        promptTokens: prompt.tokens.count, generatedTokens: tokens.count,
                        firstTextSeconds: Self.seconds(start.duration(to: firstText)),
                        elapsedSeconds: Self.seconds(start.duration(to: end)),
                        omittedTurns: prompt.omittedTurns, reachedContextLimit: tokens.count == available
                    )
                    updated.turns.append(.init(user: message, assistant: reply, metrics: metrics))
                    try ConversationStore.local().save(updated)
                    conversation = updated
                    savedChatID = updated.id.uuidString
                } catch is CancellationError {
                    draft = message
                } catch {
                    draft = message
                    self.error = error.localizedDescription
                }
            }
        } catch { self.error = error.localizedDescription }
    }
}

#endif
