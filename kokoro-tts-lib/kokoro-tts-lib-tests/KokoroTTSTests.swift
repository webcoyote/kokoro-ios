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
        The years around 1999s were wild. Prices: $12.5, £100, and 3.14. Also, initials A.B. c.
        This is most outrageous!
        """
        
        let phonemizedText = try kokoroTTS.phonemize(text: text, using: .af)
        
        let output =
        "həlˈoʊ, sˈɛd mˈɪstɚ smˈɪθ «ænd mˈɪsɪz smˈɪθ». kˈɔːl mˌiːj æt sˈɛvən ˈoʊ fˈaɪv ɔːɹ twˈɛlv əklˈɑːk. ðə jˈɪɹz ɚɹˈaʊnd nˈaɪntiːn nˈaɪndi nˈaɪnz wɜː wˈaɪld. pɹˈaɪsᵻz: twˈɛlv dˈɑːlɚz ænd fˈɪfti sˈɛnts, wˈʌn hˈʌndɹɪd pˈaʊndz, ænd θɹˈiː pˈɔɪnt wˈʌn fˈɔːɹ. ˈɔːlsoʊ, ɪnˈɪʃəlz ˈeɪbˈiː sˈiː. ðɪs ɪz mˈoʊst aʊtɹˈeɪdʒəs!"
        #expect(phonemizedText == output)
    }
}

