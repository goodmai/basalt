import Testing
import Foundation
import BasaltCore

@Suite("Token and byte counts are distinct units that cost nothing and encode as plain numbers")
struct UnitsTests {
    @Test("Units add, subtract, scale and compare within one unit, and take integer literals")
    func arithmetic() {
        let prompt: TokenCount = 1_000
        let reply = TokenCount(24)
        #expect(prompt + reply == 1_024)
        #expect(prompt - reply < prompt)
        #expect(ByteCount(512) * 4 == 2_048)
        #expect(TokenCount.zero == 0)
        #expect([prompt, reply].max() == prompt)
    }

    @Test("A unit encodes and decodes as its bare number, so DTOs keep their JSON shape")
    func codable() throws {
        let data = try JSONEncoder().encode([ByteCount(4_096)])
        #expect(String(decoding: data, as: UTF8.self) == "[4096]")
        #expect(try JSONDecoder().decode([TokenCount].self, from: Data("[7]".utf8)) == [7])
    }

    @Test("TokenCount and ByteCount are different types even though both wrap integers")
    func distinctTypes() {
        #expect(TokenCount.self != Tagged<ByteUnit, Int>.self)
        #expect(MemoryLayout<TokenCount>.size == MemoryLayout<Int>.size)
        #expect(MemoryLayout<ByteCount>.size == MemoryLayout<Int64>.size)
    }
}
