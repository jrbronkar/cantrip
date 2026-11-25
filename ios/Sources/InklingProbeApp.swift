#if os(macOS)
import Foundation

@main
struct InklingProbeCLI {
    static func main() async {
        let flags = CommandLine.arguments.dropFirst()
        let intervalFlag = "--async-layer-interval="
        let intervalOptions = flags.filter { $0.hasPrefix(intervalFlag) }
        let interval = intervalOptions.first.map { Int($0.dropFirst(intervalFlag.count)) ?? 0 }
        let cacheFlag = "--cache-mib="
        let cacheOptions = flags.filter { $0.hasPrefix(cacheFlag) }
        let cacheMiB = cacheOptions.first.map { Int($0.dropFirst(cacheFlag.count)) ?? -1 }
        let prefillFlag = "--prefill="
        let prefillOptions = flags.filter { $0.hasPrefix(prefillFlag) }
        let prefill = prefillOptions.first.map { String($0.dropFirst(prefillFlag.count)) } ?? "batched"
        let arguments = flags.filter {
            $0 != "--fingerprint-cache" && !$0.hasPrefix(intervalFlag)
                && !$0.hasPrefix(cacheFlag) && !$0.hasPrefix(prefillFlag)
        }
        let residency = arguments.count == 2
            ? InklingTokenProfile.LayerResidency.none
            : arguments.count == 3 && arguments[2].hasPrefix("--")
                ? InklingTokenProfile.LayerResidency(rawValue: arguments[2]) : nil
        guard let residency, intervalOptions.count <= 1, cacheOptions.count <= 1, prefillOptions.count <= 1,
              ["batched", "serial", "serial-all-heads"].contains(prefill),
              cacheMiB.map({ (0...1024).contains($0) }) ?? true,
              interval.map({ residency == .allAsync && (1...48).contains($0) }) ?? true else {
            FileHandle.standardError.write(Data(
                "Usage: InklingProbe CHECKPOINT_DIRECTORY FIXTURE_DIRECTORY [--resident | --resident-async | --all-resident | --all-resident-async] [--fingerprint-cache] [--async-layer-interval=1...48 for --all-resident-async] [--cache-mib=N] [--prefill=batched|serial|serial-all-heads]\n".utf8))
            exit(2)
        }
        do {
            try await InklingTokenProfile.fullDecode(
                directory: URL(fileURLWithPath: arguments[0], isDirectory: true),
                fixtureDirectory: URL(fileURLWithPath: arguments[1], isDirectory: true),
                residency: residency, fingerprintCache: flags.contains("--fingerprint-cache"),
                asyncLayerInterval: interval ?? InklingTokenProfile.defaultAsyncLayerInterval,
                cacheMiB: cacheMiB ?? InklingTokenProfile.defaultLayerCacheMiB,
                serialPrefill: prefill != "batched", predictEachPromptToken: prefill == "serial-all-heads"
            ) { FileHandle.standardOutput.write(Data(($0 + "\n").utf8)) }
        } catch {
            FileHandle.standardError.write(Data("INKLING_FULL_DECODE_ERROR \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }
}
#else
import MLX
import SwiftUI
import UniformTypeIdentifiers

@main
struct InklingProbeApp: App {
    var body: some Scene {
        WindowGroup { ProbeView() }
    }
}

struct ProbeView: View {
    @State private var importing = false
    @State private var showingChat = false
    @State private var chatMode = InklingChatMode.standard
    private enum ImportKind { case fixture, expert, checkpoint }

    @State private var importKind = ImportKind.fixture
    @State private var checkpointReport: EmbeddingProbe.CheckpointReport?
    private struct ExpertComparison {
        let cpu: EmbeddingProbe.ExpertReport
        let metal: EmbeddingProbe.ExpertReport
    }

    @State private var expertReport: ExpertComparison?
    @State private var nextTokenReport: NextTokenReport?

    private struct NextTokenReference: Decodable {
        let version: Int
        let inputIDs: [Int32]
        let expectedTokenID: Int32
        let expectedText: String
    }

    private struct NextTokenReport {
        let tokenID: Int32
        let reference: NextTokenReference
        let elapsed: Duration
        let peakMLXBytes: Int
    }

    @State private var running = false
    @State private var report: EmbeddingProbe.Report?
    @State private var error: String?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    if let directory = stagedCheckpoint, FileManager.default.fileExists(
                        atPath: directory.appendingPathComponent("inkling-full-decode/manifest.json").path
                    ) {
                        Button("Chat with Inkling 2 Mini", systemImage: "bubble.left.and.bubble.right") {
                            chatMode = .standard
                            showingChat = true
                        }
                        .disabled(running)
                        Button("Chat in Recording Mode", systemImage: "record.circle") {
                            chatMode = .recording
                            showingChat = true
                        }
                        .disabled(running)
                    }
                    if let directory = stagedCheckpoint, FileManager.default.fileExists(
                        atPath: InklingChatMode.shortContext.fixture(in: directory)
                            .appendingPathComponent("manifest.json").path
                    ) {
                        Button("Chat with Short Context", systemImage: "bubble.left") {
                            chatMode = .shortContext
                            showingChat = true
                        }
                        .disabled(running)
                    }
                    if let directory = stagedCheckpoint, FileManager.default.fileExists(
                        atPath: InklingChatMode.extendedContext.fixture(in: directory)
                            .appendingPathComponent("manifest.json").path
                    ) {
                        Button("Chat with 1K Context", systemImage: "bubble.left.and.bubble.right") {
                            chatMode = .extendedContext
                            showingChat = true
                        }
                        .disabled(running)
                    }
                    Button("Import parity fixture", systemImage: "folder") {
                        importKind = .fixture
                        importing = true
                    }
                    .disabled(running)
                    Button("Test 3.9 GB checkpoint expert", systemImage: "cpu") {
                        importKind = .expert
                        importing = true
                    }
                    .disabled(running)
                    Button("Scan full 3.9 GB checkpoint", systemImage: "internaldrive") {
                        importKind = .checkpoint
                        importing = true
                    }
                    .disabled(running)
                    if let stagedCheckpoint {
                        Button("Compare CPU and Metal expert", systemImage: "cpu") {
                            importKind = .expert
                            Task { await checkExpert(.success(stagedCheckpoint)) }
                        }
                        .disabled(running)
                        Button("Scan checkpoint copied from Mac", systemImage: "iphone") {
                            importKind = .checkpoint
                            Task { await checkCheckpoint(.success(stagedCheckpoint)) }
                        }
                        .disabled(running)
                    }
                    if let directory = stagedCheckpoint, FileManager.default.fileExists(
                        atPath: directory.appendingPathComponent("inkling-next-token.mlxfn").path
                    ) {
                        Button("Predict one offline token", systemImage: "text.cursor") {
                            Task { await predictToken(in: directory) }
                        }
                        .disabled(running)
                    }
                    if running {
                        ProgressView(importKind == .checkpoint ? "Reading checkpoint pages" : "Checking model outputs")
                    }
                    if let error { Text(error).foregroundStyle(.red) }
                } footer: {
                    if report == nil && !running && error == nil {
                        Text("Choose a reference fixture from Files to check Inkling 2 Mini offline.")
                    }
                }
                if let nextTokenReport {
                    Section("Full model next token") {
                        let reference = nextTokenReport.reference
                        Label(nextTokenReport.tokenID == reference.expectedTokenID ? "Token matched" : "Token differed",
                              systemImage: nextTokenReport.tokenID == reference.expectedTokenID ? "checkmark.circle" : "xmark.circle")
                        LabeledContent("Predicted token ID", value: String(nextTokenReport.tokenID))
                        LabeledContent("Mac reference", value: "\(reference.expectedTokenID) · \(reference.expectedText)")
                        LabeledContent("Import and prediction time", value: Self.seconds(nextTokenReport.elapsed))
                        LabeledContent("Peak MLX allocations", value: Self.byteCount(nextTokenReport.peakMLXBytes))
                    }
                }
                if let checkpointReport {
                    Section("Full checkpoint scan") {
                        LabeledContent("Weight shards", value: checkpointReport.shardCount.formatted())
                        LabeledContent("Mapped weights", value: Self.byteCount(checkpointReport.mappedBytes))
                        LabeledContent("Pages read", value: checkpointReport.pagesRead.formatted())
                        LabeledContent("Map time", value: Self.seconds(checkpointReport.mapTime))
                        LabeledContent("Read time", value: Self.seconds(checkpointReport.readTime))
                        LabeledContent("Process peak resident memory", value: Self.byteCount(checkpointReport.peakResidentBytes))
                        LabeledContent("Page sample checksum", value: String(checkpointReport.checksum, radix: 16))
                    }
                }
                if let expertReport {
                    Section("Packed expert parity") {
                        let passed = expertReport.cpu.projection.passed && expertReport.metal.projection.passed
                        Label(passed ? "CPU and Metal passed" : "Expert parity failed",
                              systemImage: passed ? "checkmark.circle" : "xmark.circle")
                        LabeledContent("CPU max absolute error", value: expertReport.cpu.projection.maxError.formatted(.number.notation(.scientific)))
                        LabeledContent("Metal max absolute error", value: expertReport.metal.projection.maxError.formatted(.number.notation(.scientific)))
                        LabeledContent("CPU elapsed", value: Self.seconds(expertReport.cpu.elapsed))
                        LabeledContent("Metal elapsed", value: Self.seconds(expertReport.metal.elapsed))
                        if let gpuSeconds = expertReport.metal.gpuSeconds {
                            LabeledContent("GPU command time", value: Self.seconds(.seconds(gpuSeconds)))
                        }
                        LabeledContent("Process peak resident memory", value: Self.byteCount(expertReport.metal.peakResidentBytes))
                    }
                }
                if let report {
                    let passed = report.embeddings.filter(\.passed).count + report.projections.filter(\.passed).count
                    let total = report.embeddings.count + report.projections.count
                    Section {
                        Label(
                            "\(passed) of \(total) checks passed",
                            systemImage: passed == total ? "checkmark.circle" : "exclamationmark.circle"
                        )
                    }
                    Section("Embedding parity") {
                        ForEach(report.embeddings) { row in
                            LabeledContent {
                                Text(row.maxError.formatted(.number.notation(.scientific)))
                                    .monospacedDigit()
                            } label: {
                                Label(
                                    "Token \(row.id) · \(row.bits)-bit",
                                    systemImage: row.passed ? "checkmark.circle" : "xmark.circle"
                                )
                                .foregroundStyle(row.passed ? Color.green : Color.red)
                            }
                            .accessibilityElement(children: .combine)
                            .accessibilityValue(row.passed ? "Passed" : "Failed")
                        }
                    }
                    if !report.projections.isEmpty {
                        Section("Layer 0 attention") {
                            ForEach(report.projections) { projection in
                                LabeledContent {
                                    Text(projection.maxError.formatted(.number.notation(.scientific)))
                                        .monospacedDigit()
                                } label: {
                                    Label(projection.id, systemImage: projection.passed ? "checkmark.circle" : "xmark.circle")
                                        .foregroundStyle(projection.passed ? Color.green : Color.red)
                                }
                                .accessibilityElement(children: .combine)
                                .accessibilityValue(projection.passed ? "Passed" : "Failed")
                            }
                        }
                    }
                }
                Section {
                    NavigationLink("Privacy") {
                        List {
                            Text("Inkling 2 Mini reads the fixture or checkpoint files you choose and computes results on this device. It does not upload those files or collect analytics.")
                            Text("The probe keeps results in memory until you import another fixture or quit. Use Files to manage the original fixture.")
                        }
                        .navigationTitle("Privacy")
                    }
                    NavigationLink("Open-source notices") {
                        ScrollView {
                            Text(Self.notice).padding()
                                .textSelection(.enabled)
                        }
                        .navigationTitle("Open-source notices")
                    }
                }
            }
            .navigationTitle("Inkling 2 Mini")
            .navigationDestination(isPresented: $showingChat) {
                if let directory = stagedCheckpoint {
                    InklingChatView(directory: directory, mode: chatMode)
                }
            }
            .task {
                let arguments = ProcessInfo.processInfo.arguments
                if arguments.contains("--bandwidth-screen") {
                    do {
                        let result = try await Task.detached(priority: .userInitiated) {
                            try EmbeddingProbe.runBandwidthScreen()
                        }.value
                        Self.logPrediction("INKLING_BANDWIDTH " + String(decoding: result, as: UTF8.self))
                    } catch {
                        Self.logPrediction("INKLING_BANDWIDTH_ERROR \(error.localizedDescription)")
                    }
                    return
                }
                if let routeMode = EmbeddingProbe.RouteMode.allCases.first(where: {
                    arguments.contains("--" + $0.rawValue)
                }) {
                    do {
                        guard let directory = stagedCheckpoint else { throw EmbeddingProbe.Failure.invalidFixture }
                        let result = try await Task.detached(priority: .userInitiated) {
                            try EmbeddingProbe.runRouteScreen(in: directory, mode: routeMode)
                        }.value
                        Self.logPrediction("INKLING_ROUTE " + String(decoding: result, as: UTF8.self))
                    } catch {
                        Self.logPrediction("INKLING_ROUTE_ERROR \(error.localizedDescription)")
                    }
                    return
                }
                if arguments.contains("--mapped-expert") || arguments.contains("--copied-expert") {
                    let mapped = arguments.contains("--mapped-expert")
                    do {
                        guard let directory = stagedCheckpoint else { throw EmbeddingProbe.Failure.invalidFixture }
                        let report = try await Task.detached(priority: .userInitiated) {
                            try EmbeddingProbe.runExpert(
                                archive: directory.appendingPathComponent("layer_00_packed.npz"),
                                reference: directory.appendingPathComponent("inkling-expert-reference.json"),
                                useMetal: true, mappedWeights: mapped, repetitions: 16)
                        }.value
                        let data = try JSONSerialization.data(withJSONObject: [
                            "mapped": mapped, "passed": report.projection.passed, "max_error": report.projection.maxError,
                            "archive_bytes": report.archiveBytes, "peak_resident_bytes": report.peakResidentBytes,
                            "gpu_seconds": report.gpuSamples, "dispatch_seconds": report.dispatchSamples,
                            "buffer_setup_seconds": report.bufferSetupSeconds ?? 0,
                            "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
                        ], options: .sortedKeys)
                        Self.logPrediction("INKLING_EXPERT " + String(decoding: data, as: UTF8.self))
                    } catch {
                        Self.logPrediction("INKLING_EXPERT_ERROR \(error.localizedDescription)")
                    }
                    return
                }
                let boundedAsync = arguments.contains("--full-decode-resident-async")
                let shortDecode = arguments.contains("--full-decode-short")
                let extendedDecode = arguments.contains("--full-decode-1024")
                let decodeMode: InklingChatMode = extendedDecode ? .extendedContext : shortDecode ? .shortContext : .standard
                let allResidentAsync = shortDecode || extendedDecode || arguments.contains("--full-decode-all-resident-async")
                if arguments.contains("--full-decode") || arguments.contains("--full-decode-resident")
                    || boundedAsync || allResidentAsync {
                    let wasIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
                    UIApplication.shared.isIdleTimerDisabled = true
                    defer { UIApplication.shared.isIdleTimerDisabled = wasIdleTimerDisabled }
                    do {
                        guard let directory = stagedCheckpoint else { throw EmbeddingProbe.Failure.invalidFixture }
                        let residency: InklingTokenProfile.LayerResidency = if allResidentAsync {
                            .allAsync
                        } else if boundedAsync {
                            .recurrentSubsetAsync
                        } else {
                            arguments.contains("--full-decode-resident") ? .recurrentSubset : .none
                        }
                        let cacheFlag = "--cache-mib="
                        var cacheMiB = InklingTokenProfile.defaultLayerCacheMiB
                        if let option = arguments.first(where: { $0.hasPrefix(cacheFlag) }) {
                            guard let value = Int(option.dropFirst(cacheFlag.count)), (0...1024).contains(value) else {
                                throw EmbeddingProbe.Failure.invalidFixture
                            }
                            cacheMiB = value
                        }
                        try await InklingTokenProfile.fullDecode(directory: directory,
                            fixtureDirectory: decodeMode.fixture(in: directory),
                            residency: residency,
                            fingerprintCache: arguments.contains("--fingerprint-cache"),
                            cacheMiB: cacheMiB, serialPrefill: shortDecode || extendedDecode, emit: Self.logPrediction)
                    } catch {
                        Self.logPrediction("INKLING_FULL_DECODE_ERROR \(error.localizedDescription)")
                    }
                    return
                }
                if arguments.contains("--layer-decode") {
                    let wasIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
                    UIApplication.shared.isIdleTimerDisabled = true
                    defer { UIApplication.shared.isIdleTimerDisabled = wasIdleTimerDisabled }
                    do {
                        guard let directory = stagedCheckpoint else { throw EmbeddingProbe.Failure.invalidFixture }
                        try await InklingTokenProfile.layerDecode(directory: directory, emit: Self.logPrediction)
                    } catch {
                        Self.logPrediction("INKLING_LAYER_DECODE_ERROR \(error.localizedDescription)")
                    }
                    return
                }
                if arguments.contains("--cached-step") || arguments.contains("--cached-step-resident") {
                    let wasIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
                    UIApplication.shared.isIdleTimerDisabled = true
                    defer { UIApplication.shared.isIdleTimerDisabled = wasIdleTimerDisabled }
                    do {
                        guard let directory = stagedCheckpoint else { throw EmbeddingProbe.Failure.invalidFixture }
                        try await InklingTokenProfile.cachedStep(directory: directory,
                            retainGraph: arguments.contains("--cached-step-resident"), emit: Self.logPrediction)
                    } catch {
                        Self.logPrediction("INKLING_CACHED_STEP_ERROR \(error.localizedDescription)")
                    }
                    return
                }
                if arguments.contains("--benchmark") {
                    let wasIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
                    UIApplication.shared.isIdleTimerDisabled = true
                    defer { UIApplication.shared.isIdleTimerDisabled = wasIdleTimerDisabled }
                    do {
                        guard let directory = stagedCheckpoint else { throw EmbeddingProbe.Failure.invalidFixture }
                        try await InklingTokenProfile.benchmark(directory: directory, emit: Self.logPrediction)
                    } catch {
                        Self.logPrediction("INKLING_BENCH_ERROR \(error.localizedDescription)")
                    }
                    return
                }
                if arguments.contains("--open-1024-chat") {
                    chatMode = .extendedContext
                    showingChat = true
                } else if arguments.contains("--open-short-chat") {
                    chatMode = .shortContext
                    showingChat = true
                } else if arguments.contains("--open-recording-chat") {
                    chatMode = .recording
                    showingChat = true
                } else if arguments.contains("--open-chat") {
                    showingChat = true
                }
                let shortChatSmoke = arguments.contains("--chat-short-smoke-test")
                let asyncChatSmoke = shortChatSmoke || arguments.contains("--chat-async-smoke-test")
                if arguments.contains("--chat-smoke-test") || asyncChatSmoke, let directory = stagedCheckpoint {
                    let wasIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
                    UIApplication.shared.isIdleTimerDisabled = true
                    defer { UIApplication.shared.isIdleTimerDisabled = wasIdleTimerDisabled }
                    do {
                        let runtime = try await InklingChatRuntime.load(
                            from: directory, useLayerDecoder: asyncChatSmoke,
                            mode: shortChatSmoke ? .shortContext : .standard)
                        if asyncChatSmoke {
                            let start = ProcessInfo.processInfo.systemUptime
                            try await runtime.warmUp()
                            Self.logPrediction("Chat warm-up seconds: \(ProcessInfo.processInfo.systemUptime - start)")
                        }
                        let prompt = try runtime.prepare(Conversation(), message: "What is 2+2?")
                        guard prompt.tokens == runtime.config.sampleInputIDs else {
                            throw EmbeddingProbe.Failure.invalidFixture
                        }
                        var ids = prompt.tokens
                        for index in 0..<4 {
                            let start = ProcessInfo.processInfo.systemUptime
                            ids.append(try await runtime.nextToken(after: ids))
                            if asyncChatSmoke && index == 0 {
                                Self.logPrediction("Chat first token after warm-up seconds: \(ProcessInfo.processInfo.systemUptime - start)")
                            }
                        }
                        let output = Array(ids.dropFirst(prompt.tokens.count))
                        guard output.first == Int(runtime.config.sampleNextTokenID) else {
                            throw EmbeddingProbe.Failure.invalidFixture
                        }
                        if asyncChatSmoke && !shortChatSmoke && arguments.contains("--chat-context-smoke-test") {
                            let context = try runtime.prepare(Conversation(), message:
                                String(repeating: " red", count: 160) + ". What color was repeated?")
                            guard context.reservedReplyTokens == 64,
                                  (176...192).contains(context.tokens.count) else {
                                throw EmbeddingProbe.Failure.invalidFixture
                            }
                            let start = ProcessInfo.processInfo.systemUptime
                            var contextIDs = context.tokens
                            while contextIDs.count < runtime.config.capacity {
                                contextIDs.append(try await runtime.nextToken(after: contextIDs))
                            }
                            Self.logPrediction("Chat context test: \(context.tokens.count) prompt tokens, "
                                + "\(contextIDs.count) total tokens in \(ProcessInfo.processInfo.systemUptime - start) seconds")
                        }
                        Self.logPrediction("Chat tokens: \(output), expected first: \(runtime.config.sampleNextTokenID)")
                        Self.logPrediction("Chat output: \(runtime.tokenizer.decode(tokens: output, skipSpecialTokens: true))")
                    } catch {
                        Self.logPrediction("Chat failed: \(error.localizedDescription)")
                    }
                }
                if ProcessInfo.processInfo.arguments.contains("--predict-one-token"), let directory = stagedCheckpoint {
                    await predictToken(in: directory)
                }
            }
            .fileImporter(isPresented: $importing, allowedContentTypes: importKind == .fixture ? [.json] : [.folder]) { selection in
                switch importKind {
                case .fixture: Task { await check(selection) }
                case .expert: Task { await checkExpert(selection) }
                case .checkpoint: Task { await checkCheckpoint(selection) }
                }
            }
        }
    }

    private var stagedCheckpoint: URL? {
        let directory = URL.documentsDirectory.appendingPathComponent("inkling-3.9gb-checkpoint", isDirectory: true)
        let manifest = directory.appendingPathComponent("manifest.json")
        return FileManager.default.fileExists(atPath: manifest.path) ? directory : nil
    }

    private static func byteCount(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    private static func seconds(_ elapsed: Duration) -> String {
        elapsed.formatted(.units(allowed: [.seconds], fractionalPart: .show(length: 3)))
    }

    @MainActor
    private func predictToken(in directory: URL) async {
        running = true
        error = nil
        nextTokenReport = nil
        defer { running = false }
        do {
            let result = try await InklingMLXWorker.shared.run {
                let graph = directory.appendingPathComponent("inkling-next-token.mlxfn")
                let referenceFile = directory.appendingPathComponent("inkling-next-token-reference.json")
                let reference = try JSONDecoder().decode(
                    NextTokenReference.self, from: EmbeddingProbe.loadFixture(from: referenceFile)
                )
                guard reference.version == 1, !reference.inputIDs.isEmpty else {
                    throw EmbeddingProbe.Failure.invalidFixture
                }
                Memory.cacheLimit = 64 * 1024 * 1024
                Memory.peakMemory = 0
                let start = ContinuousClock.now
                Self.logPrediction("Importing the full MLX graph")
                let tokenID = try InklingMLXWorker.greedyToken(graph: graph, inputs: [reference.inputIDs])
                return NextTokenReport(tokenID: tokenID, reference: reference,
                                       elapsed: start.duration(to: .now), peakMLXBytes: Memory.peakMemory)
            }
            nextTokenReport = result
            Self.logPrediction("Token \(result.tokenID), expected \(result.reference.expectedTokenID), elapsed \(result.elapsed), peak MLX bytes \(result.peakMLXBytes)")
        } catch {
            self.error = error.localizedDescription
            Self.logPrediction("Prediction failed: \(error.localizedDescription)")
        }
    }

    private nonisolated static func logPrediction(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    @MainActor
    private func checkCheckpoint(_ selection: Result<URL, Error>) async {
        running = true
        error = nil
        checkpointReport = nil
        defer { running = false }
        do {
            let directory = try selection.get()
            checkpointReport = try await Task.detached(priority: .userInitiated) {
                let scoped = directory.startAccessingSecurityScopedResource()
                defer { if scoped { directory.stopAccessingSecurityScopedResource() } }
                return try EmbeddingProbe.scanCheckpoint(in: directory)
            }.value
        } catch {
            self.error = error.localizedDescription
        }
    }

    @MainActor
    private func checkExpert(_ selection: Result<URL, Error>) async {
        running = true
        error = nil
        expertReport = nil
        defer { running = false }
        do {
            let directory = try selection.get()
            expertReport = try await Task.detached(priority: .userInitiated) {
                let scoped = directory.startAccessingSecurityScopedResource()
                defer { if scoped { directory.stopAccessingSecurityScopedResource() } }
                let archive = directory.appendingPathComponent("layer_00_packed.npz")
                let reference = directory.appendingPathComponent("inkling-expert-reference.json")
                return try ExpertComparison(
                    cpu: EmbeddingProbe.runExpert(archive: archive, reference: reference),
                    metal: EmbeddingProbe.runExpert(archive: archive, reference: reference, useMetal: true)
                )
            }.value
        } catch {
            self.error = error.localizedDescription
        }
    }

    private static let notice: String = {
        guard let url = Bundle.main.url(forResource: "LookAtMyQuant-MIT", withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            return "LookAtMyQuant · MIT License"
        }
        return "LookAtMyQuant\n\n" + text
    }()

    @MainActor
    private func check(_ selection: Result<URL, Error>) async {
        running = true
        error = nil
        report = nil
        defer { running = false }
        do {
            let url = try selection.get()
            report = try await Task.detached(priority: .userInitiated) {
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                return try EmbeddingProbe.run(EmbeddingProbe.loadFixture(from: url))
            }.value
        } catch {
            self.error = error.localizedDescription
        }
    }
}

#endif
