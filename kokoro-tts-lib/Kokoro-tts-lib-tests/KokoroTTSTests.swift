//
//  kokoro-tts-lib
//
import Testing
@testable import KokoroTTSLib
import CoreML
import Foundation

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
    
    @Test func loadVoicePack() throws {
        let kokoroTTS = try KokoroTTS()
        let voiceData = try kokoroTTS.voiceData(for: .af)
        #expect(voiceData != nil)
    }
    
    @Test func testTTS() throws {
        let tokens = [50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]
                
        let kokoroTTS = try KokoroTTS()
        let bertModelOutput: MLMultiArray
        let textMask: MLMultiArray
        let totalTokenCount: Int
                
        (bertModelOutput, _, textMask, totalTokenCount) = try kokoroTTS.bertModel.forwardPassOnInitialModel(tokens: tokens)
        let forwardModelOutput = try kokoroTTS.bertModel.forwardPassOnEncoderModel(array: bertModelOutput)
                
        let textEncoderOutput = try kokoroTTS.prosodyPredictorEngine.executeTextEncoder(
            input: forwardModelOutput,
            referenceVoice: try kokoroTTS.voiceData(for: .af),
            inputLength: Int32(tokens.count),
            textMask: textMask)
        
        let lstmOutput = try kokoroTTS.prosodyPredictorEngine.executeLSTM(input: textEncoderOutput)
        let durationProj = try kokoroTTS.prosodyPredictorEngine.executeDurationProj(input: lstmOutput)
        let duration = try kokoroTTS.prosodyPredictorEngine.calculateDuration(input: durationProj)
        
        print(textEncoderOutput.shape)
        print(lstmOutput.shape)
        print(durationProj.shape)
        print(duration.shape)
        
        let roundAndClamp: MLMultiArray
        let totalDuration: Int
        (roundAndClamp, totalDuration) = try kokoroTTS.prosodyPredictorEngine.roundAndClamp(input: duration)
        
        print("TOTAL DUR ", totalDuration)
        
        let en = try kokoroTTS.prosodyPredictorEngine.processAlignment(
            inputLengths: totalTokenCount,
            predDurSum: totalDuration,
            predDur: roundAndClamp,
            textEncoderOutput: textEncoderOutput)
        print(en.shape)
    }
}

