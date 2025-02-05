//
//  kokoro-tts-lib
//
import Testing
@testable import KokoroTTSLib

struct ESpeakNGTests {
    @Test func testInitialization() throws {
        let eSpeakNG = try ESpeakNGEngine()
        #expect(eSpeakNG != nil)
    }
    
    @Test func testSetLanguage() throws {
        let eSpeakNG = try ESpeakNGEngine()
        try eSpeakNG.setLanguage(.enUS)
    }
    
    @Test func testPhonemize() throws {
        let eSpeakNG = try ESpeakNGEngine()
        
        let text = TextNormalizer.normalizeText(
        """
        ‘Hello’, said Mr. Smith (and Mrs. Smith). Call me at 7:05 or 12:00.
        The year 1999s was wild. Prices: $12.5, £100, and 3.14. Also, initials A.B. c.
        """)
                
        try eSpeakNG.setLanguage(.enUS)
        let phonemizedText = try eSpeakNG.phonemize(text: text)
        
        print(phonemizedText)

        #expect(phonemizedText != nil)
    }
}

