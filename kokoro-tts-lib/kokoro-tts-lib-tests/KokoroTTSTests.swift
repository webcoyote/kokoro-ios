//
//  kokoro-tts-lib
//
import Testing
@testable import KokoroTTSLib

struct KokoroTTSTests {
    @Test func testPhonemize() throws {
        let kokoroTTS = try KokoroTTS()
        
        let text =
        """
        ‘Hello’, said Mr. Smith (and Mrs. Smith). Call me at 7:05 or 12:00.
        The year 1999s was wild. Prices: $12.5, £100, and 3.14. Also, initials A.B. c.
        """
        
        let phonemizedText = try kokoroTTS.phonemize(text: text, using: .af)
        print(phonemizedText)
        
        #expect(phonemizedText != nil)
    }
}

