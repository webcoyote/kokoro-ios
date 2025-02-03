//
//  kokoro-tts-lib
//
import Testing
@testable import KokoroTTSLib

struct ESpeakNGTests {
    @Test func testInitialization() async throws {
        var eSpeakNG = try ESpeakNGEngine()
        #expect(eSpeakNG != nil)
    }
}

