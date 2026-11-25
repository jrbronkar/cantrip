// Packed embedding and attention projection algorithms adapted from LookAtMyQuant (MIT).
// Copyright (c) 2026 Shin Megami Boson. See Notices/LookAtMyQuant-MIT.txt.
import Darwin
import Foundation
import Metal

struct EmbeddingFixture: Decodable {
    struct Row: Decodable {
        let token: Int
        let bits: Int
        let words: [UInt32]
        let scale: Float
        let table: [Float]
        let signs: [Float]
        let expected: [Float]
    }

    let version: Int
    let epsilon: Float
    let norm: [Float]
    let rows: [Row]
    let projections: ProjectionFixture?
}

struct ProjectionFixture: Decodable {
    struct Matrix: Decodable {
        let name: String
        let words: Data
        let signs: [Float]
        let scales: [Float]
        let expected: [Float]
    }

    let token: Int
    let norm: [Float]
    let table: [Float]
    let matrices: [Matrix]
    let attention: Attention?

    struct Attention: Decodable {
        let qScale: [Float]
        let relScale: [Float]
        let qkClip: Float
        let relClip: Float
        let keyTap: [Float]
        let valueTap: [Float]
        let relativeZero: [Float]
        let sinks: [Float]
        let expectedAttention: [Float]
        let output: Matrix
        let postTap: [Float]
        let expectedPost: [Float]
    }
}

enum EmbeddingProbe {
    static let maxFixtureBytes = 16 * 1024 * 1024
    static let tolerance: Float = 0.0001
    private static let width = 2304
    private static let columns = 72
    private static let paley72 = paley(columns)
    private static let paley12 = paley(12)

    private static func paley(_ columns: Int) -> [Float] {
        let prime = columns - 1
        let squares = Set((1..<prime).map { $0 * $0 % prime })
        return (0..<(columns * columns)).map { index in
            let row = index / columns
            let col = index % columns
            if row == col || row == 0 { return 1 }
            if col == 0 { return -1 }
            return squares.contains((col - row + prime) % prime) ? 1 : -1
        }
    }

    struct Result: Identifiable {
        let id: Int
        let bits: Int
        let maxError: Float

        var passed: Bool { maxError <= EmbeddingProbe.tolerance }
    }

    enum Failure: LocalizedError {
        case invalidFixture
        case fixtureTooLarge

        var errorDescription: String? {
            switch self {
            case .invalidFixture: return "Choose a valid Inkling parity fixture."
            case .fixtureTooLarge: return "The fixture exceeds the 16 MiB size limit."
            }
        }
    }

    struct ProjectionResult: Identifiable {
        let id: String
        let maxError: Float

        var passed: Bool { maxError <= 0.002 }
    }

    struct Report {
        let embeddings: [Result]
        let projections: [ProjectionResult]
    }

    static func loadFixture(from url: URL) throws -> Data {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var data = Data()
        while let chunk = try file.read(upToCount: min(64 * 1024, maxFixtureBytes + 1 - data.count)),
              !chunk.isEmpty {
            data.append(chunk)
            guard data.count <= maxFixtureBytes else { throw Failure.fixtureTooLarge }
        }
        return data
    }

    static func run(_ data: Data) throws -> Report {
        let fixture: EmbeddingFixture
        do {
            fixture = try JSONDecoder().decode(EmbeddingFixture.self, from: data)
        } catch is DecodingError {
            throw Failure.invalidFixture
        }
        guard fixture.version == 1, fixture.norm.count == width,
              fixture.epsilon.isFinite, fixture.epsilon > 0,
              (1...32).contains(fixture.rows.count),
              Set(fixture.rows.map(\.token)).count == fixture.rows.count else {
            throw Failure.invalidFixture
        }
        var projectionResults: [ProjectionResult] = []
        let embeddingResults = try fixture.rows.map { row in
            guard [1, 2, 3, 6].contains(row.bits), row.words.count == width * row.bits / 32,
                  row.table.count == (row.bits == 6 ? 64 : 4096),
                  row.signs.count == width, row.expected.count == width else {
                throw Failure.invalidFixture
            }
            let output = decode(row, norm: fixture.norm, epsilon: fixture.epsilon)
            guard output.allSatisfy(\.isFinite) else { throw Failure.invalidFixture }
            if let projections = fixture.projections, projections.token == row.token {
                projectionResults = try project(output, fixture: projections, epsilon: fixture.epsilon)
            }
            let error = zip(output, row.expected).map { abs($0 - $1) }.max() ?? .infinity
            return Result(id: row.token, bits: row.bits, maxError: error)
        }
        guard fixture.projections == nil || !projectionResults.isEmpty else { throw Failure.invalidFixture }
        return Report(embeddings: embeddingResults, projections: projectionResults)
    }

    private static func project(
        _ embedding: [Float], fixture: ProjectionFixture, epsilon: Float
    ) throws -> [ProjectionResult] {
        let shapes = [3072, 1024, 1024, 384, 3072]
        guard fixture.norm.count == width, fixture.table.count == 2048,
              fixture.matrices.count == shapes.count,
              Set(fixture.matrices.map(\.name)).count == shapes.count else {
            throw Failure.invalidFixture
        }
        let normalized = normalize(embedding, weights: fixture.norm, epsilon: epsilon)
        let projections = try zip(fixture.matrices, shapes).map { matrix, count in
            try projectMatrix(matrix, input: normalized, outputCount: count, table: fixture.table)
        }
        var results = projections.map(\.result)
        if let attention = fixture.attention {
            results += try checkAttention(
                attention, projections: projections.map(\.output),
                normalized: normalized, table: fixture.table, epsilon: epsilon
            )
        }
        return results
    }

    private static func projectMatrix(
        _ matrix: ProjectionFixture.Matrix, input: [Float], outputCount: Int, table: [Float]
    ) throws -> (result: ProjectionResult, output: [Float]) {
        let inputCount = input.count
        let wordsPerRow = inputCount * 3 / 32
        guard matrix.words.count == outputCount * wordsPerRow * 4, matrix.signs.count == inputCount,
              matrix.scales.count == outputCount, matrix.expected.count == outputCount else {
            throw Failure.invalidFixture
        }
        var signed = zip(input, matrix.signs).map { $0 * $1 }
        let denominator = sqrt(Float(inputCount))
        let rotated = transform(&signed, inverse: false).map { Float(Float16($0 / denominator)) }
        let output = matrix.words.withUnsafeBytes { bytes in
            (0..<outputCount).map { row in
                let start = row * wordsPerRow
                var dot: Float = 0
                // HT3 hashes each reversed 12-bit window into two adjacent coefficients.
                for i in Swift.stride(from: 0, to: inputCount, by: 2) {
                    let bit = (inputCount - 2 - i) * 3
                    let word = bit / 32
                    let shift = bit % 32
                    let offset = (start + word) * 4
                    let low = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) >> shift
                    let high = shift > 20 && word + 1 < wordsPerRow
                        ? UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self)) << (32 - shift) : 0
                    let state = (low | high) & 4095
                    let index = 2 * Int((state &* 2_654_435_761) >> 22)
                    dot += table[index] * rotated[i] + table[index + 1] * rotated[i + 1]
                }
                return dot * matrix.scales[row]
            }
        }
        return (try projectionResult(matrix.name, output, matrix.expected), output)
    }

    private static func checkAttention(
        _ fixture: ProjectionFixture.Attention, projections: [[Float]],
        normalized: [Float], table: [Float], epsilon: Float
    ) throws -> [ProjectionResult] {
        let headWidth = 128
        let relWidth = 16
        let heads = 24
        let query = projections[0], key = projections[1], value = projections[2]
        let relative = projections[3], gate = projections[4]
        guard fixture.qScale.count == query.count, fixture.relScale.count == relative.count,
              fixture.keyTap.count == key.count, fixture.valueTap.count == value.count,
              fixture.qkClip > 0, fixture.relClip > 0, fixture.relativeZero.count == relWidth,
              fixture.sinks.count == heads * width, fixture.expectedAttention.count == query.count,
              fixture.postTap.count == width, fixture.expectedPost.count == width else {
            throw Failure.invalidFixture
        }
        let q = clipHeads(zip(query, fixture.qScale).map(*), width: headWidth, limit: fixture.qkClip)
        let r = clipHeads(zip(relative, fixture.relScale).map(*), width: relWidth, limit: fixture.relClip)
        let k = clipHeads(firstConvolution(key, taps: fixture.keyTap), width: headWidth, limit: fixture.qkClip)
            .map { Float(Float16($0)) }
        let convolvedV = firstConvolution(value, taps: fixture.valueTap)
        let v = Swift.stride(from: 0, to: value.count, by: headWidth).flatMap { start in
            let head = convolvedV[start..<(start + headWidth)]
            let inverse = 1 / sqrt(dot(head, head) / Float(headWidth) + epsilon)
            return head.map { Float(Float16($0 * inverse)) }
        }
        var attention: [Float] = []
        for head in 0..<heads {
            let qStart = head * headWidth
            let kvStart = (head / 3) * headWidth
            let relStart = head * relWidth
            let sinkStart = head * width
            let bias = dot(r[relStart..<(relStart + relWidth)], fixture.relativeZero)
            let score = dot(q[qStart..<(qStart + headWidth)], k[kvStart..<(kvStart + headWidth)])
                / sqrt(Float(headWidth)) + bias
            let sink = dot(normalized, fixture.sinks[sinkStart..<(sinkStart + width)])
            let probability = 1 / (1 + exp(sink - score))
            attention.append(contentsOf: v[kvStart..<(kvStart + headWidth)].map { $0 * probability })
        }
        let gated = zip(attention, gate).map { $0 / (1 + exp(-$1)) }
        let output = try projectMatrix(fixture.output, input: gated, outputCount: width, table: table)
        let post = firstConvolution(output.output, taps: fixture.postTap)
        return [
            try projectionResult("Grouped-query attention", attention, fixture.expectedAttention),
            output.result,
            try projectionResult("Post-attention convolution", post, fixture.expectedPost),
        ]
    }

    private static func projectionResult(
        _ label: String, _ output: [Float], _ expected: [Float]
    ) throws -> ProjectionResult {
        guard output.allSatisfy(\.isFinite) else { throw Failure.invalidFixture }
        let error = zip(output, expected).map { abs($0 - $1) }.max() ?? .infinity
        return ProjectionResult(id: label, maxError: error)
    }

    private static func dot<A: Sequence, B: Sequence>(_ a: A, _ b: B) -> Float
    where A.Element == Float, B.Element == Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    private static func firstConvolution(_ values: [Float], taps: [Float]) -> [Float] {
        // At position zero all three previous convolution inputs are zero.
        zip(values, taps).map { ($0 + $0 * $1) / sqrt(5) }
    }

    private static func clipHeads(_ values: [Float], width: Int, limit: Float) -> [Float] {
        Swift.stride(from: 0, to: values.count, by: width).flatMap { start in
            let head = values[start..<(start + width)]
            let scale = min(limit / (sqrt(dot(head, head)) + 1e-6), 1)
            return head.map { $0 * scale }
        }
    }

    private static func decode(_ row: EmbeddingFixture.Row, norm: [Float], epsilon: Float) -> [Float] {
        // LM6 stores forward 6-bit indices. Row trellis uses reverse 12-bit windows.
        var values = (0..<width).map { index in
            let isLM6 = row.bits == 6
            let bit = (isLM6 ? index : width - 1 - index) * row.bits
            let word = bit / 32
            let shift = bit % 32
            let low = row.words[word] >> shift
            let high = shift > 0 && word + 1 < row.words.count
                ? row.words[word + 1] << (32 - shift) : 0
            let code = Int((low | high) & (isLM6 ? 63 : 4095))
            return row.table[code] * row.scale
        }

        let transformed = zip(transform(&values, inverse: true), row.signs).map { $0 * ($1 / 48) }
        return normalize(transformed, weights: norm, epsilon: epsilon)
    }

    private static func transform(_ values: inout [Float], inverse: Bool) -> [Float] {
        let columns = values.count == width ? Self.columns : 12
        let paley = columns == Self.columns ? paley72 : paley12
        let rows = values.count / columns
        var stride = 1
        while stride < rows {
            for start in Swift.stride(from: 0, to: rows, by: 2 * stride) {
                for i in 0..<(stride * columns) {
                    let a = start * columns + i
                    let b = a + stride * columns
                    let left = values[a]
                    let right = values[b]
                    values[a] = left + right
                    values[b] = left - right
                }
            }
            stride *= 2
        }

        return values.indices.map { index in
            let base = index / columns * columns
            let column = index % columns
            var sum: Float = 0
            for k in 0..<columns {
                let offset = inverse ? k * columns + column : column * columns + k
                sum += values[base + k] * paley[offset]
            }
            return sum
        }
    }

    private static func normalize(_ transformed: [Float], weights: [Float], epsilon: Float) -> [Float] {
        let meanSquare = transformed.reduce(0) { $0 + $1 * $1 } / Float(width)
        let invRMS = 1 / sqrt(meanSquare + epsilon)
        return zip(transformed, weights).map { $0 * invRMS * $1 }
    }
}


extension EmbeddingProbe {
    struct ExpertReport {
        let projection: ProjectionResult
        let archiveBytes: Int
        let elapsed: Duration
        let gpuSeconds: Double?
        let gpuSamples: [Double]
        let dispatchSamples: [Double]
        let bufferSetupSeconds: Double?
        let peakResidentBytes: Int
    }

    private struct ExpertReference: Decodable {
        let version: Int
        let archiveBytes: Int
        let offsets: [Int]
        let table: [Float]
        let expected: [Float]
    }

    static func runExpert(archive: URL, reference: URL, useMetal: Bool = false, mappedWeights: Bool = false,
                          repetitions: Int = 1) throws -> ExpertReport {
        let reference = try JSONDecoder().decode(ExpertReference.self, from: loadFixture(from: reference))
        let wordsPerRow = width / 32
        let locations = 128
        let groupCount = width / locations
        let sizes = [width * locations * wordsPerRow * 4, width * locations * 2,
                     width * 2, locations * groupCount * 4]
        guard reference.version == 1, reference.offsets.count == sizes.count,
              reference.table.count == 4096, reference.table.allSatisfy(\.isFinite),
              reference.expected.count == width, reference.expected.allSatisfy(\.isFinite),
              reference.archiveBytes >= sizes[0],
              zip(reference.offsets, sizes).allSatisfy({ offset, size in
                  offset >= 0 && offset <= reference.archiveBytes - size
              }) else { throw Failure.invalidFixture }

        let start = ContinuousClock.now
        let packed = try Data(contentsOf: archive, options: .alwaysMapped)
        guard packed.count == reference.archiveBytes else { throw Failure.invalidFixture }

        var gpuSamples: [Double] = []
        var dispatchSamples: [Double] = []
        var bufferSetupSeconds: Double?
        let output = try packed.withUnsafeBytes { bytes -> [Float] in
            let codes = reference.offsets[0]
            let scales = reference.offsets[1]
            let signs = reference.offsets[2]
            let groups = reference.offsets[3]

            func half(at offset: Int) -> Float {
                let bits = UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
                return Float(Float16(bitPattern: bits))
            }

            var input = (0..<width).map { index in
                (Float(index % 31) - 15) / 16 * half(at: signs + index * 2)
            }
            let rotated = transform(&input, inverse: false).map { $0 / 48 }
            let rowLocations = try (0..<groupCount).map { group in
                let entry = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: groups + group * 4, as: UInt32.self))
                let location = Int(entry >> 16)
                guard entry & 0xffff == 3, location < locations else { throw Failure.invalidFixture }
                return location
            }

            if mappedWeights {
                guard codes.isMultiple(of: 4), scales.isMultiple(of: 2) else {
                    throw Failure.invalidFixture
                }
                let layout = [UInt32(codes / 4), UInt32(scales / 2)] + rowLocations.map(UInt32.init)
                let result = try metalExpert(input: rotated, table: reference.table, repetitions: repetitions,
                                             mappedWeights: true) { device in
                    let weights = try mappedExpertBuffer(archive: archive, size: packed.count, device: device)
                    return [weights, try expertBuffer(layout, device: device)]
                }
                gpuSamples = result.samples
                dispatchSamples = result.dispatch
                bufferSetupSeconds = result.bufferSetup
                return result.values
            }
            if useMetal {
                var words: [UInt32] = []
                words.reserveCapacity(width * wordsPerRow)
                var rowScales: [Float] = []
                rowScales.reserveCapacity(width)
                for row in 0..<width {
                    let slot = row * locations + rowLocations[row / locations]
                    let rowStart = codes + slot * wordsPerRow * 4
                    for word in 0..<wordsPerRow {
                        words.append(UInt32(littleEndian: bytes.loadUnaligned(
                            fromByteOffset: rowStart + word * 4, as: UInt32.self
                        )))
                    }
                    rowScales.append(half(at: scales + slot * 2))
                }
                let result = try metalExpert(input: rotated, table: reference.table, repetitions: repetitions) { device in
                    [try expertBuffer(words, device: device), try expertBuffer(rowScales, device: device)]
                }
                gpuSamples = result.samples
                dispatchSamples = result.dispatch
                bufferSetupSeconds = result.bufferSetup
                return result.values
            }

            return (0..<width).map { row in
                let slot = row * locations + rowLocations[row / locations]
                let rowStart = codes + slot * wordsPerRow * 4
                var dot: Float = 0
                for word in 0..<wordsPerRow {
                    let offset = rowStart + word * 4
                    let low = UInt64(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
                    let high = word + 1 < wordsPerRow
                        ? UInt64(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self))) : 0
                    let window = low | (high << 32)
                    for shift in 0..<32 {
                        let index = width - 1 - (word * 32 + shift)
                        dot += reference.table[Int((window >> shift) & 4095)] * rotated[index]
                    }
                }
                return dot * half(at: scales + slot * 2)
            }
        }
        let result = try projectionResult("Layer 0 expert 0 gate/up", output, reference.expected)
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        return ExpertReport(projection: result, archiveBytes: packed.count, elapsed: start.duration(to: .now),
                            gpuSeconds: gpuSamples.first, gpuSamples: gpuSamples, dispatchSamples: dispatchSamples,
                            bufferSetupSeconds: bufferSetupSeconds, peakResidentBytes: Int(usage.ru_maxrss))
    }

    // Adapted from LookAtMyQuant's reverse-window trellis matvec (MIT).
    private static let expertKernel = """
    #include <metal_stdlib>
    using namespace metal;

    inline float trellisDot(device const uint *words, device const float *input,
                            device const float *table, uint lane) {
        constexpr uint wordsPerRow = 72;
        constexpr uint width = wordsPerRow * 32;
        float sum = 0.0f;
        for (uint word = 0; word < wordsPerRow; ++word) {
            uint state = words[word] >> lane;
            if (lane > 20 && word + 1 < wordsPerRow) {
                state |= words[word + 1] << (32 - lane);
            }
            sum += table[state & 4095u] * input[width - 1 - (word * 32 + lane)];
        }
        return simd_sum(sum);
    }

    kernel void packedExpert(
        device const uint *codes [[buffer(0)]],
        device const float *scales [[buffer(1)]],
        device const float *input [[buffer(2)]],
        device const float *table [[buffer(3)]],
        device float *output [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        float total = trellisDot(codes + row * 72u, input, table, lane);
        if (lane == 0) {
            output[row] = total * scales[row];
        }
    }

    kernel void mappedExpert(
        device const uint *archive [[buffer(0)]],
        device const uint *layout [[buffer(1)]],
        device const float *input [[buffer(2)]],
        device const float *table [[buffer(3)]],
        device float *output [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        uint slot = row * 128u + layout[2u + row / 128u];
        device const half *scales = reinterpret_cast<device const half *>(archive);
        float total = trellisDot(archive + layout[0] + slot * 72u, input, table, lane);
        if (lane == 0) {
            output[row] = total * float(scales[layout[1] + slot]);
        }
    }
    kernel void routedMapped(
        device const uint *archive [[buffer(0)]],
        device const uint *layout [[buffer(1)]],
        device const float *input [[buffer(2)]],
        device const float *table [[buffer(3)]],
        device float *output [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        uint source = layout[2u + row / 128u] * 128u + row % 128u;
        device const half *scales = reinterpret_cast<device const half *>(archive);
        float total = trellisDot(archive + layout[0] + source * 72u, input, table, lane);
        if (lane == 0) {
            output[row] = total * float(scales[layout[1] + source]);
        }
    }

    // Use the native T=1 kernel's half-SIMD, four-row scheduling with the
    // probe's FP32 input and scalar RTR1 lookup for a matched comparison.
    kernel void routedHalfSIMD(
        device const uint *archive [[buffer(0)]],
        device const uint *layout [[buffer(1)]],
        device const float *input [[buffer(2)]],
        device const float *table [[buffer(3)]],
        device float *output [[buffer(4)]],
        uint gridThread [[thread_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        uint row = (gridThread / 16u) * 4u;
        uint halfLane = lane % 16u;
        uint source = layout[2u + row / 128u] * 128u + row % 128u;
        device const uint *codes = archive + layout[0] + source * 72u;
        device const half *scales = reinterpret_cast<device const half *>(archive) + layout[1] + source;
        float sums[4] = {};
        for (uint word = 0; word < 72u; ++word) {
            float low = input[2303u - word * 32u - halfLane];
            float high = input[2287u - word * 32u - halfLane];
            #pragma unroll
            for (uint r = 0; r < 4u; ++r) {
                device const uint *words = codes + r * 72u;
                uint packed = words[word];
                uint upper = packed >> (halfLane + 16u);
                if (halfLane > 4u && word + 1u < 72u) {
                    upper |= words[word + 1u] << (16u - halfLane);
                }
                sums[r] += table[(packed >> halfLane) & 4095u] * low;
                sums[r] += table[upper & 4095u] * high;
            }
        }
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            float total = sums[r];
            for (ushort offset = 8; offset > 0; offset >>= 1) {
                total += simd_shuffle_xor(total, offset);
            }
            if (halfLane == 0u) {
                output[row + r] = total * float(scales[r]);
            }
        }
    }

    // Group three adjacent trellis states per lookup while retaining FP32 arithmetic.
    kernel void routedGroupedRTR1(
        device const uint *archive [[buffer(0)]],
        device const uint *layout [[buffer(1)]],
        device const float *input [[buffer(2)]],
        device const float4 *table [[buffer(3)]],
        device float *output [[buffer(4)]],
        uint gridThread [[thread_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        uint row = (gridThread / 16u) * 4u;
        uint halfLane = lane % 16u;
        uint source = layout[2u + row / 128u] * 128u + row % 128u;
        device const uint *codes = archive + layout[0] + source * 72u;
        device const half *scales = reinterpret_cast<device const half *>(archive) + layout[1] + source;
        float sums[4] = {};
        for (uint chunk = halfLane; chunk < 144u; chunk += 16u) {
            uint column = chunk * 16u;
            uint bit = 2304u - column - 16u;
            uint word = bit / 32u;
            float4 x0 = *reinterpret_cast<device const float4 *>(input + column);
            float4 x1 = *reinterpret_cast<device const float4 *>(input + column + 4u);
            float4 x2 = *reinterpret_cast<device const float4 *>(input + column + 8u);
            float4 x3 = *reinterpret_cast<device const float4 *>(input + column + 12u);
            #pragma unroll
            for (uint r = 0; r < 4u; ++r) {
                device const uint *words = codes + r * 72u;
                uint window = words[word];
                if ((bit & 16u) != 0u) {
                    uint next = word + 1u < 72u ? words[word + 1u] : 0u;
                    window = (window >> 16u) | (next << 16u);
                }
                float total = dot(table[(window >> 13u) & 16383u].xyz, x0.xyz);
                total += dot(table[(window >> 10u) & 16383u].xyz, float3(x0.w, x1.xy));
                total += dot(table[(window >> 7u) & 16383u].xyz, float3(x1.zw, x2.x));
                total += dot(table[(window >> 4u) & 16383u].xyz, x2.yzw);
                total += dot(table[(window >> 1u) & 16383u].xyz, x3.xyz);
                sums[r] += total + table[window & 16383u].z * x3.w;
            }
        }
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            float total = sums[r];
            for (ushort offset = 8; offset > 0; offset >>= 1) {
                total += simd_shuffle_xor(total, offset);
            }
            if (halfLane == 0u) {
                output[row + r] = total * float(scales[r]);
            }
        }
    }
    """

    private static let metalExpertUnavailable = NSError(domain: "InklingMetal", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "Could not initialize Metal for the packed expert."
    ])

    struct RouteScreen: Decodable {
        struct Sample: Decodable {
            let expert: Int
            let row: Int
            let expected: Float
        }
        struct Layer: Decodable {
            let archiveBytes: Int
            let offsets: [Int]
            let rounds: [[Sample]]
        }
        let version: Int
        let table: [Float]
        let layers: [Layer]
    }

    enum RouteMode: String, CaseIterable {
        case staged = "route-screen"
        case mapped = "route-screen-mapped"
        case halfSIMD = "route-screen-half-simd"
        case halfSIMD128 = "route-screen-half-simd-128"
        case groupedRTR1 = "route-screen-grouped-rtr1"

        var threads: Int {
            switch self {
            case .staged, .mapped: 32
            case .halfSIMD, .groupedRTR1: 256
            case .halfSIMD128: 128
            }
        }

        var kernel: String {
            switch self {
            case .staged: "packedExpert"
            case .mapped: "routedMapped"
            case .halfSIMD, .halfSIMD128: "routedHalfSIMD"
            case .groupedRTR1: "routedGroupedRTR1"
            }
        }
    }

    static func runRouteScreen(in directory: URL, mode: RouteMode = .staged) throws -> Data {
        let fixture = try JSONDecoder().decode(RouteScreen.self, from: loadFixture(
            from: directory.appendingPathComponent("inkling-route-screen.json")))
        let slots = 8
        let locations = 128
        let rows = width * slots
        let rowBytes = width / 8
        let groups = width / locations
        let sizes = [width * locations * rowBytes, width * locations * 2, width * 2, locations * groups * 4]
        guard fixture.version == 1, fixture.table.count == 4096, fixture.layers.count == 6,
              fixture.table.allSatisfy(\.isFinite),
              fixture.layers.allSatisfy({ layer in
                  layer.offsets.count == sizes.count && layer.archiveBytes >= sizes[0] &&
                  zip(layer.offsets, sizes).allSatisfy { offset, size in
                      offset >= 0 && offset <= layer.archiveBytes - size
                  } &&
                  layer.rounds.count == 16 && layer.rounds.allSatisfy { route in
                      route.count == slots && Set(route.map(\.expert)).count == slots &&
                      route.allSatisfy { (0..<locations).contains($0.expert) && (0..<width).contains($0.row)
                          && $0.expected.isFinite }
                  }
              }) else { throw Failure.invalidFixture }
        let unavailable = metalExpertUnavailable
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let codes = device.makeBuffer(length: rows * rowBytes, options: .storageModeShared),
              let scales = device.makeBuffer(length: rows * 4, options: .storageModeShared),
              let output = device.makeBuffer(length: rows * 4, options: .storageModeShared) else { throw unavailable }
        let library = try device.makeLibrary(source: expertKernel, options: nil)
        guard let function = library.makeFunction(name: mode.kernel) else { throw unavailable }
        let pipeline = try device.makeComputePipelineState(function: function)
        let threads = mode.threads
        let rowsPerGroup = threads == 32 ? 1 : threads / 16 * 4
        let threadgroups = rows / rowsPerGroup
        guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= threads else {
            throw unavailable
        }
        let table: MTLBuffer
        if mode == .groupedRTR1 {
            let grouped = (0..<16384).map { state in
                SIMD4<Float>(fixture.table[state >> 2], fixture.table[(state >> 1) & 4095],
                             fixture.table[state & 4095], 0)
            }
            table = try expertBuffer(grouped, device: device)
        } else {
            table = try expertBuffer(fixture.table, device: device)
        }
        let layers = try fixture.layers.enumerated().map { index, metadata in
            let archive = directory.appendingPathComponent(String(format: "layer_%02d_packed.npz", index))
            let packed = try Data(contentsOf: archive, options: .alwaysMapped)
            guard packed.count == metadata.archiveBytes else { throw Failure.invalidFixture }
            let rotated = packed.withUnsafeBytes { bytes in
                var input = (0..<width).map { index in
                    let bits = UInt16(littleEndian: bytes.loadUnaligned(
                        fromByteOffset: metadata.offsets[2] + index * 2, as: UInt16.self))
                    return (Float(index % 31) - 15) / 16 * Float(Float16(bitPattern: bits))
                }
                return transform(&input, inverse: false).map { $0 / 48 }
            }
            var mapped: MTLBuffer?
            if mode != .staged {
                guard metadata.offsets[0].isMultiple(of: 4), metadata.offsets[1].isMultiple(of: 2) else {
                    throw Failure.invalidFixture
                }
                mapped = try mappedExpertBuffer(archive: archive, size: packed.count, device: device)
            }
            return (packed, try expertBuffer(rotated, device: device), mapped)
        }
        let layout = try expertBuffer([UInt32](repeating: 0, count: 2 + slots * groups), device: device)
        let layoutPointer = layout.contents().bindMemory(to: UInt32.self, capacity: 2 + slots * groups)
        let codesPointer = codes.contents()
        let scalesPointer = scales.contents().bindMemory(to: Float.self, capacity: rows)
        let outputPointer = output.contents().bindMemory(to: Float.self, capacity: rows)
        var staging: [[Double]] = []
        var dispatch: [[Double]] = []
        var gpu: [[Double]] = []
        var maxError: Float = 0
        let initialThermal = ProcessInfo.processInfo.thermalState.rawValue
        for round in 0..<16 {
            var stagingRound: [Double] = []
            var dispatchRound: [Double] = []
            var gpuRound: [Double] = []
            for (index, layer) in fixture.layers.enumerated() {
                try autoreleasepool {
                    let start = ProcessInfo.processInfo.systemUptime
                    let route = layer.rounds[round]
                    let (packed, input, mapped) = layers[index]
                    layoutPointer[0] = UInt32(layer.offsets[0] / 4)
                    layoutPointer[1] = UInt32(layer.offsets[1] / 2)
                    try packed.withUnsafeBytes { bytes in
                        guard let base = bytes.baseAddress else { throw Failure.invalidFixture }
                        for (slot, sample) in route.enumerated() {
                            for group in 0..<groups {
                                let groupOffset = layer.offsets[3] + (sample.expert * groups + group) * 4
                                let entry = UInt32(littleEndian: bytes.loadUnaligned(
                                    fromByteOffset: groupOffset, as: UInt32.self))
                                let location = Int(entry >> 16)
                                guard entry & 0xffff == 3, location < width else { throw Failure.invalidFixture }
                                if mapped != nil {
                                    layoutPointer[2 + slot * groups + group] = UInt32(location)
                                    continue
                                }
                                let source = location * locations
                                let destination = slot * width + group * locations
                                let codeOffset = layer.offsets[0] + source * rowBytes
                                codesPointer.advanced(by: destination * rowBytes).copyMemory(
                                    from: base.advanced(by: codeOffset), byteCount: locations * rowBytes)
                                for row in 0..<locations {
                                    let scaleOffset = layer.offsets[1] + (source + row) * 2
                                    let bits = UInt16(littleEndian: bytes.loadUnaligned(
                                        fromByteOffset: scaleOffset, as: UInt16.self))
                                    scalesPointer[destination + row] = Float(Float16(bitPattern: bits))
                                }
                            }
                        }
                    }
                    let staged = ProcessInfo.processInfo.systemUptime
                    guard let command = queue.makeCommandBuffer(),
                          let encoder = command.makeComputeCommandEncoder() else { throw unavailable }
                    encoder.setComputePipelineState(pipeline)
                    let weightBuffers = mapped.map { [$0, layout] } ?? [codes, scales]
                    for (slot, buffer) in (weightBuffers + [input, table, output]).enumerated() {
                        encoder.setBuffer(buffer, offset: 0, index: slot)
                    }
                    encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    guard command.status == .completed else { throw command.error ?? unavailable }
                    let end = ProcessInfo.processInfo.systemUptime
                    stagingRound.append(staged - start)
                    dispatchRound.append(end - staged)
                    gpuRound.append(command.gpuEndTime - command.gpuStartTime)
                    for (slot, sample) in route.enumerated() {
                        let actual = outputPointer[slot * width + sample.row]
                        guard actual.isFinite else { throw Failure.invalidFixture }
                        maxError = max(maxError, abs(actual - sample.expected))
                    }
                }
            }
            staging.append(stagingRound)
            dispatch.append(dispatchRound)
            gpu.append(gpuRound)
        }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        return try JSONSerialization.data(withJSONObject: [
            "staging_seconds": staging, "dispatch_seconds": dispatch, "gpu_seconds": gpu,
            "mode": mode.rawValue, "threads_per_group": threads,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "max_error": maxError, "passed": maxError <= 0.0001,
            "staging_bytes": mode == .staged ? codes.length + scales.length : layout.length,
            "peak_resident_bytes": usage.ru_maxrss, "initial_thermal": initialThermal,
            "final_thermal": ProcessInfo.processInfo.thermalState.rawValue,
        ], options: .sortedKeys)
    }

    static func runBandwidthScreen() throws -> Data {
        let length = 64 * 1024 * 1024
        let unavailable = metalExpertUnavailable
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let source = device.makeBuffer(length: length, options: .storageModePrivate),
              let destination = device.makeBuffer(length: length, options: .storageModePrivate),
              let setup = queue.makeCommandBuffer(), let fill = setup.makeBlitCommandEncoder() else {
            throw unavailable
        }
        fill.fill(buffer: source, range: 0..<length, value: 123)
        fill.endEncoding()
        setup.commit()
        setup.waitUntilCompleted()
        guard setup.status == .completed else { throw setup.error ?? unavailable }

        let initialThermal = ProcessInfo.processInfo.thermalState.rawValue
        let initialLowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        let samples = try (0..<16).map { _ in
            try autoreleasepool {
                guard let command = queue.makeCommandBuffer(),
                      let copy = command.makeBlitCommandEncoder() else { throw unavailable }
                copy.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: length)
                copy.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                guard command.status == .completed else { throw command.error ?? unavailable }
                return command.gpuEndTime - command.gpuStartTime
            }
        }
        return try JSONSerialization.data(withJSONObject: [
            "bytes_read": length, "bytes_written": length, "gpu_seconds": samples,
            "initial_thermal": initialThermal, "final_thermal": ProcessInfo.processInfo.thermalState.rawValue,
            "initial_low_power": initialLowPower,
            "final_low_power": ProcessInfo.processInfo.isLowPowerModeEnabled,
        ], options: .sortedKeys)
    }

    private static func expertBuffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let buffer = device.makeBuffer(bytes: base, length: bytes.count, options: .storageModeShared) else {
                throw metalExpertUnavailable
            }
            return buffer
        }
    }

    private static func mappedExpertBuffer(archive: URL, size: Int, device: MTLDevice) throws -> MTLBuffer {
        let pageSize = Int(getpagesize())
        let length = (size + pageSize - 1) / pageSize * pageSize
        guard length <= device.maxBufferLength else { throw metalExpertUnavailable }
        let file = try FileHandle(forReadingFrom: archive)
        defer { try? file.close() }
        var info = stat()
        guard fstat(file.fileDescriptor, &info) == 0, info.st_size == size else { throw Failure.invalidFixture }
        // A private mapping permits Metal to wire the pages without writing to the checkpoint.
        let address = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, file.fileDescriptor, 0)
        guard let address, address != MAP_FAILED else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        guard let buffer = device.makeBuffer(bytesNoCopy: address, length: length, options: .storageModeShared,
                                             deallocator: { pointer, size in _ = munmap(pointer, size) }) else {
            munmap(address, length)
            throw NSError(domain: "InklingMetal", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Metal refused a shared buffer backed by the checkpoint mapping."
            ])
        }
        buffer.label = "Mapped Inkling expert archive"
        return buffer
    }

    private static func metalExpert(
        input: [Float], table: [Float], repetitions: Int, mappedWeights: Bool = false,
        weights: (MTLDevice) throws -> [MTLBuffer]
    ) throws -> (values: [Float], samples: [Double], dispatch: [Double], bufferSetup: Double) {
        let unavailable = metalExpertUnavailable
        guard repetitions > 0, let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { throw unavailable }
        let library = try device.makeLibrary(source: expertKernel, options: nil)
        guard let function = library.makeFunction(name: mappedWeights ? "mappedExpert" : "packedExpert") else {
            throw unavailable
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        let setupStart = ProcessInfo.processInfo.systemUptime
        let inputs = try weights(device) + [expertBuffer(input, device: device), expertBuffer(table, device: device)]
        guard pipeline.threadExecutionWidth == 32,
              let output = device.makeBuffer(length: width * MemoryLayout<Float>.stride,
                                             options: .storageModeShared) else { throw unavailable }
        var samples: [Double] = []
        var dispatch: [Double] = []
        let bufferSetup = ProcessInfo.processInfo.systemUptime - setupStart
        for _ in 0..<repetitions {
            let start = ProcessInfo.processInfo.systemUptime
            try autoreleasepool {
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { throw unavailable }
                encoder.setComputePipelineState(pipeline)
                for (index, input) in inputs.enumerated() {
                    encoder.setBuffer(input, offset: 0, index: index)
                }
                encoder.setBuffer(output, offset: 0, index: inputs.count)
                encoder.dispatchThreadgroups(MTLSize(width: width, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                guard command.status == .completed else { throw command.error ?? unavailable }
                let elapsed = command.gpuEndTime - command.gpuStartTime
                if elapsed > 0 { samples.append(elapsed) }
            }
            dispatch.append(ProcessInfo.processInfo.systemUptime - start)
        }
        let values = output.contents().bindMemory(to: Float.self, capacity: width)
        return (Array(UnsafeBufferPointer(start: values, count: width)), samples, dispatch, bufferSetup)
    }

}

extension EmbeddingProbe {
    struct CheckpointReport {
        let shardCount: Int
        let mappedBytes: Int
        let pagesRead: Int
        let checksum: UInt64
        let mapTime: Duration
        let readTime: Duration
        let peakResidentBytes: Int
    }

    static func scanCheckpoint(in directory: URL) throws -> CheckpointReport {
        let layers = (0..<24).map { String(format: "layer_%02d_packed.npz", $0) }
        let names = ["embed_packed.npz", "head_packed.npz", "luts.npz", "steering.safetensors"] + layers
        let start = ContinuousClock.now
        let shards = try names.map { name in
            let file = try FileHandle(forReadingFrom: directory.appendingPathComponent(name))
            defer { try? file.close() }
            var info = stat()
            guard fstat(file.fileDescriptor, &info) == 0 else { throw POSIXError(.EIO) }
            guard (info.st_mode & S_IFMT) == S_IFREG, info.st_size > 0 else {
                throw CocoaError(.fileReadCorruptFile)
            }
            let count = Int(info.st_size)
            let address = mmap(nil, count, PROT_READ, MAP_PRIVATE, file.fileDescriptor, 0)
            guard let address, address != MAP_FAILED else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            return Data(bytesNoCopy: address, count: count, deallocator: .unmap)
        }
        let mapped = ContinuousClock.now
        let pageSize = Int(getpagesize())
        var pagesRead = 0
        var checksum: UInt64 = 0
        for shard in shards {
            shard.withUnsafeBytes { bytes in
                for offset in stride(from: 0, to: bytes.count, by: pageSize) {
                    checksum = (checksum &* 1_099_511_628_211) ^ UInt64(bytes[offset])
                    pagesRead += 1
                }
            }
        }
        let finished = ContinuousClock.now
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        return CheckpointReport(shardCount: shards.count, mappedBytes: shards.reduce(0) { $0 + $1.count },
                                pagesRead: pagesRead, checksum: checksum, mapTime: start.duration(to: mapped),
                                readTime: mapped.duration(to: finished), peakResidentBytes: Int(usage.ru_maxrss))
    }
}
