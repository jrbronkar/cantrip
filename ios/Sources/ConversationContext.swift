import Foundation

struct Conversation: Codable, Identifiable, Equatable {
    struct Turn: Codable, Equatable {
        struct Metrics: Codable, Equatable {
            let promptTokens: Int
            let generatedTokens: Int
            let firstTextSeconds: Double
            let elapsedSeconds: Double
            let omittedTurns: Int
            let reachedContextLimit: Bool

            var tokensPerSecond: Double {
                Double(generatedTokens) / max(elapsedSeconds, 0.001)
            }
        }

        var user: String
        var assistant: String
        var metrics: Metrics? = nil
    }

    let id: UUID
    var systemPrompt: String?
    var turns: [Turn]

    init(id: UUID = UUID(), systemPrompt: String? = nil, turns: [Turn] = []) {
        self.id = id
        self.systemPrompt = systemPrompt
        self.turns = turns
    }
}

struct ConversationStore {
    let directory: URL

    static func local() throws -> Self {
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true
        )
        return Self(directory: support.appendingPathComponent("Inkling/Conversations", isDirectory: true))
    }

    func save(_ conversation: Conversation) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        #if os(iOS)
        let options: Data.WritingOptions = [.atomic, .completeFileProtection]
        #else
        let options: Data.WritingOptions = .atomic
        #endif
        try JSONEncoder().encode(conversation).write(to: url(for: conversation.id), options: options)
    }

    func load(_ id: UUID) throws -> Conversation {
        try JSONDecoder().decode(Conversation.self, from: Data(contentsOf: url(for: id)))
    }

    func list() throws -> [UUID] {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles]
        )
        return files.filter { $0.pathExtension == "json" }
            .compactMap { file -> (id: UUID, modified: Date)? in
                guard let id = UUID(uuidString: file.deletingPathExtension().lastPathComponent) else { return nil }
                let modified = (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
                return (id, modified ?? .distantPast)
            }
            .sorted { left, right in
                left.modified == right.modified ? left.id.uuidString < right.id.uuidString : left.modified > right.modified
            }
            .map(\.id)
    }

    private func url(for id: UUID) -> URL {
        directory.appendingPathComponent(id.uuidString).appendingPathExtension("json")
    }
}

enum ConversationContext {
    struct Prepared {
        let tokens: [Int]
        let omittedTurns: Int
        let reservedReplyTokens: Int

        /// Reuse only when the live cache covers an exact prefix of the new prompt.
        /// The caller scopes the cache to the same model, tokenizer and conversation.
        func reusablePrefixLength(cached: [Int]) -> Int {
            guard cached.count <= tokens.count, tokens.starts(with: cached) else { return 0 }
            return cached.count
        }
    }

    enum Failure: LocalizedError {
        case blankMessage
        case reservedMarker
        case invalidBudget
        case promptTooLong

        var errorDescription: String? {
            switch self {
            case .blankMessage: return "Enter a message before sending."
            case .reservedMarker: return "The conversation contains an Inkling control marker."
            case .invalidBudget: return "The context limit must leave room for both prompt and reply."
            case .promptTooLong: return "The system instructions and latest message exceed the available context."
            }
        }
    }

    static func prepare(
        _ conversation: Conversation, nextUserMessage: String,
        contextLimit: Int, replyReserve: Int, encode: (String) throws -> [Int]
    ) throws -> Prepared {
        guard replyReserve > 0, contextLimit > replyReserve else { throw Failure.invalidBudget }
        guard !nextUserMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Failure.blankMessage
        }
        let prefix = try conversation.systemPrompt.map { try message("system", $0) } ?? ""
        let latest = try message("user", nextUserMessage) + "<|message_model|><|content_text|>"
        let effort = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        let history = try conversation.turns.map { turn in
            try message("user", turn.user) + message("model", turn.assistant) + "<|content_model_end_sampling|>"
        }
        let promptLimit = contextLimit - replyReserve
        for omitted in 0...history.count {
            let rendered = prefix + effort + history.dropFirst(omitted).joined() + latest
            let tokens = try encode(rendered)
            if tokens.count <= promptLimit {
                return Prepared(tokens: tokens, omittedTurns: omitted, reservedReplyTokens: replyReserve)
            }
        }
        throw Failure.promptTooLong
    }

    private static func message(_ role: String, _ text: String) throws -> String {
        // Inkling's tokenizer treats these literal markers as structural tokens.
        guard text.range(of: #"<\|[^|<>]+\|>"#, options: .regularExpression) == nil else {
            throw Failure.reservedMarker
        }
        return "<|message_\(role)|><|content_text|>\(text)<|end_message|>"
    }
}
