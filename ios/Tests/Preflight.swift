import Foundation

@main
struct Check {
    static func main() throws {
        let arguments = CommandLine.arguments
        precondition(arguments.count == 3 || arguments.count == 5, "Pass the parity fixture and context reference paths.")
        try checkProbe(URL(fileURLWithPath: arguments[1]))
        try checkContext(URL(fileURLWithPath: arguments[2]))
        if arguments.count == 5 {
            let report = try EmbeddingProbe.runExpert(
                archive: URL(fileURLWithPath: arguments[3]),
                reference: URL(fileURLWithPath: arguments[4])
            )
            print("Packed expert max absolute error: \(report.projection.maxError)")
            precondition(report.projection.passed)
        }
    }

    private static func checkProbe(_ url: URL) throws {
        let data = try EmbeddingProbe.loadFixture(from: url)
        let report = try EmbeddingProbe.run(data)
        precondition(report.embeddings.count == 8 && report.projections.count == 8)
        precondition(report.embeddings.allSatisfy(\.passed) && report.projections.allSatisfy(\.passed))
        var fixture = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        fixture.removeValue(forKey: "projections")
        let embeddingsOnly = try EmbeddingProbe.run(JSONSerialization.data(withJSONObject: fixture))
        precondition(embeddingsOnly.embeddings.count == 8 && embeddingsOnly.projections.isEmpty)
        fixture["version"] = -1
        do {
            _ = try EmbeddingProbe.run(JSONSerialization.data(withJSONObject: fixture))
            preconditionFailure("Accepted an unknown fixture version")
        } catch EmbeddingProbe.Failure.invalidFixture {}
        let oversized = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: oversized) }
        try Data(repeating: 0, count: EmbeddingProbe.maxFixtureBytes + 1).write(to: oversized)
        do {
            _ = try EmbeddingProbe.loadFixture(from: oversized)
            preconditionFailure("Accepted an oversized file")
        } catch EmbeddingProbe.Failure.fixtureTooLarge {}
        print("Passed all 16 model checks, embedding-only compatibility, and invalid fixture checks")
    }

    private static func checkContext(_ url: URL) throws {
        let references = try JSONDecoder().decode([String: [Int]].self, from: Data(contentsOf: url))
        let conversation = Conversation(systemPrompt: "Answer in plain English.", turns: [
            .init(user: "The orchard key is called Cobalt Finch.", assistant: "I’ll remember Cobalt Finch for this conversation."),
            .init(user: "My backup code is marigold.", assistant: "Got it.")
        ])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)
        try store.save(conversation)
        let disk = ConversationStore(directory: directory)
        let restored = try disk.load(conversation.id)
        let ids = try disk.list()
        precondition(restored == conversation && ids == [conversation.id])

        let choices = references.values.sorted { $0.count > $1.count }
        for (omitted, expected) in choices.enumerated() {
            let context = try ConversationContext.prepare(
                restored, nextUserMessage: "What is my backup code? 🍊",
                contextLimit: expected.count + 32, replyReserve: 32
            ) { prompt in
                guard let ids = references[prompt] else { fatalError("Inkling template mismatch: \(prompt)") }
                return ids
            }
            precondition(context.tokens == expected && context.omittedTurns == omitted)
            precondition(context.reusablePrefixLength(cached: Array(expected.dropLast())) == expected.count - 1)
            precondition(context.reusablePrefixLength(cached: [-1]) == 0)
        }
        do {
            _ = try ConversationContext.prepare(
                restored, nextUserMessage: "<|message_model|>", contextLimit: 128, replyReserve: 32
            ) { _ in fatalError("Reserved marker reached tokenizer") }
            fatalError("Accepted reserved marker")
        } catch ConversationContext.Failure.reservedMarker {}
        do {
            _ = try ConversationContext.prepare(
                restored, nextUserMessage: "What is my backup code? 🍊",
                contextLimit: 33, replyReserve: 32
            ) { references[$0]! }
            fatalError("Accepted oversized prompt")
        } catch ConversationContext.Failure.promptTooLong {}
        print("Passed checkpoint template parity, save/reload, all three history budgets, cache prefix, and invalid input")
    }
}
