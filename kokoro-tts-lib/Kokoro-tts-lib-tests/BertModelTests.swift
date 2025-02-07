//
//  kokoro-tts-lib
//
import Testing
import CoreML
@testable import KokoroTTSLib

struct BertModelTests {
    @Test func testBertModel() throws {
        let tokens = [50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]
        
        let bertModel = try BertModel()
        let prosodyPredictorEngine = try ProsodyPredictorEngine()
        let kokoroTTS = try KokoroTTS()

        let bertModelOutput: MLMultiArray
        let textMask: MLMultiArray
        let tokenCount: Int

        (bertModelOutput, _, textMask, tokenCount) = try bertModel.forwardPassOnInitialModel(tokens: tokens)
        let forwardModelOutput = try bertModel.forwardPassOnEncoderModel(array: bertModelOutput)
        
        let textEncoderOutput = try prosodyPredictorEngine.executeTextEncoder(
            input: forwardModelOutput,
            referenceVoice: try kokoroTTS.voiceData(for: .af),
            inputLength: Int32(tokenCount - 2),
            textMask: textMask)
        
        let lstmOutput = try prosodyPredictorEngine.executeLSTM(input: textEncoderOutput)
        print(lstmOutput.shape)
    }
}

