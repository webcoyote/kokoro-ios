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
        The years around 1999s were wild. Prices: $12.5, £100, and 3.14. Also, initials A.B. c.
        This is most outrageous!
        """)
        
        print(text)
                
        #expect(text ==
        """
        'Hello', said Mister Smith «and Mrs Smith». Call me at 7 oh 5 or 12 o'clock.
         The years around 19 99s were wild. Prices: 12 dollars and 50 cents, 100 pounds, and 3 point 1 4. Also, initials A-B- c.
         This is most outrageous!
        """)
    }
    
    @Test func testTokenize() throws {
        let phonemizedText = "həlˈoʊ, sˈɛd mˈɪstɚ smˈɪθ «ænd mˈɪsɪz smˈɪθ». kˈɔːl mˌiː æt sˈɛvən ˈoʊ fˈaɪv ɔːɹ twˈɛlv əklˈɑːk. ðə jˈɪɹ nˈaɪntiːn nˈaɪndi nˈaɪnz wʌz wˈaɪld. pɹˈaɪsᵻz: twˈɛlv dˈɑːlɚz ænd fˈɪfti sˈɛnts, wˈʌn hˈʌndɹɪd pˈaʊndz, ænd θɹˈiː pˈɔɪnt wˈʌn fˈɔːɹ. ˈɔːlsoʊ, ɪnˈɪʃəlz ˈeɪbˈiː sˈiː."
        
        let outputArray = TextNormalizer.tokenize(phonemizedText)
                
        let expectedArray: [Int] = [50, 83, 54, 156, 57, 135, 3, 16, 61, 156, 86, 46, 16, 55, 156, 102, 61, 62, 85, 16, 61, 55, 156, 102, 119, 16, 12, 72, 56, 46, 16, 55, 156, 102, 61, 102, 68, 16, 61, 55, 156, 102, 119, 13, 4, 16, 53, 156, 76, 158, 54, 16, 55, 157, 51, 158, 16, 72, 62, 16, 61, 156, 86, 64, 83, 56, 16, 156, 57, 135, 16, 48, 156, 43, 102, 64, 16, 76, 158, 123, 16, 62, 65, 156, 86, 54, 64, 16, 83, 53, 54, 156, 69, 158, 53, 4, 16, 81, 83, 16, 52, 156, 102, 123, 16, 56, 156, 43, 102, 56, 62, 51, 158, 56, 16, 56, 156, 43, 102, 56, 46, 51, 16, 56, 156, 43, 102, 56, 68, 16, 65, 138, 68, 16, 65, 156, 43, 102, 54, 46, 4, 16, 58, 123, 156, 43, 102, 61, 177, 68, 2, 16, 62, 65, 156, 86, 54, 64, 16, 46, 156, 69, 158, 54, 85, 68, 16, 72, 56, 46, 16, 48, 156, 102, 48, 62, 51, 16, 61, 156, 86, 56, 62, 61, 3, 16, 65, 156, 138, 56, 16, 50, 156, 138, 56, 46, 123, 102, 46, 16, 58, 156, 43, 135, 56, 46, 68, 3, 16, 72, 56, 46, 16, 119, 123, 156, 51, 158, 16, 58, 156, 76, 102, 56, 62, 16, 65, 156, 138, 56, 16, 48, 156, 76, 158, 123, 4, 16, 156, 76, 158, 54, 61, 57, 135, 3, 16, 102, 56, 156, 102, 131, 83, 54, 68, 16, 156, 47, 102, 44, 156, 51, 158, 16, 61, 156, 51, 158, 4]
                
        #expect(outputArray.count == expectedArray.count)
        
        expectedArray.enumerated().forEach { index, value in
            print(index)
            #expect(outputArray[index] == expectedArray[index])
        }
    }
}
