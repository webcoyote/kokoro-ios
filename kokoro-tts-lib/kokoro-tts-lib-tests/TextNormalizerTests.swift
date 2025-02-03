//
//  kokoro-tts-lib-tests
//
import Testing
@testable import KokoroTTSLib

struct TextNormalizerTests {
    @Test func testNormalization() async throws {
        let text = TextNormalizer.normalizeText(
        """
        ‘Hello’, said Mr. Smith (and Mrs. Smith). Call me at 7:05 or 12:00.
        The year 1999s was wild. Prices: $12.5, £100, and 3.14. Also, initials A.B. c.
        """)
                
        #expect(text ==
        """
        'Hello', said Mister Smith «and Mrs Smith». Call me at 7 oh 5 or 12 o'clock.
        The year 19 99s was wild. Prices: 12 dollars and 50 cents, 100 pounds, and 3 point 1 4. Also, initials A-B- c.
        """)
    }
}
