//
//  kokoro-tts-lib
//
import Testing
import CoreML
@testable import KokoroTTSLib

struct BertModelTests {
    static let tokens = [50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]
        
    let testBundle = Bundle(for: ProsodyPredictorEngineTestsBundleCapturer.self)
    let bertModel: BertModel
    
    init() throws {
        bertModel = try BertModel()
    }

    @Test func testForwardPassOnInitialModel() throws {
        let tokensWithPadding = [0] + BertModelTests.tokens + [0]
        let tokensCount = tokensWithPadding.count
            
        let tokensShape: [NSNumber] = [1, NSNumber(value: tokensCount)]
        let tokensArray = try MLMultiArray(shape: tokensShape, dataType: .int32)
            
        for i in 0..<tokensCount {
            tokensArray[[0, NSNumber(value: i)]] = NSNumber(value: tokensWithPadding[i])
        }
            
        let inputLengths = [tokensCount]
        let textMask = try bertModel.lengthToMask(lengths: inputLengths)
        
        let tokensRealOutput = try! MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "tokens", shape: [1, 143], dataType: .int32)
        let textMaskRealOutput = try! MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "text_mask", shape: [1,143], dataType: .int32)
        
        #expect(tokensArray == tokensRealOutput)
        #expect(textMask == textMaskRealOutput)
        
        let inputs = bert_modelInput(input_ids: tokensArray, mask: textMask)
        let output: bert_modelOutput = try bertModel.initialModel.prediction(input: inputs)
        
        let realOutput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "bert_dur", shape: [1, 143, 768])
        let closeEnough = MLMultiArray.allClose(output.sequence_output, realOutput!, rtol: 1e-4, atol: 1e-4)
        #expect(closeEnough)
    }
    
    @Test func testEncoderModel() throws {
        let realInput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "bert_dur", shape: [1, 143, 768])!
        let realOutput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "d_en", shape: [1, 512, 143])!
        
        let output = try bertModel.forwardPassOnEncoderModel(array: realInput)
        
        let closeEnough = MLMultiArray.allClose(output, realOutput, rtol: 1e-4, atol: 1e-4)
        #expect(closeEnough)
    }
    
    @Test func testTotalBertModel() throws {
        let bertModelOutput: MLMultiArray
        let textMask: MLMultiArray
        let totalTokenCount: Int

        (bertModelOutput, _, textMask, totalTokenCount) = try bertModel.forwardPassOnInitialModel(tokens: BertModelTests.tokens)
        let forwardModelOutput = try bertModel.forwardPassOnEncoderModel(array: bertModelOutput)
             
        let textMaskRealOutput = try! MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "text_mask", shape: [1,143], dataType: .int32)
        let realOutput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "d_en", shape: [1, 512, 143])!

        #expect(totalTokenCount == 143)
        #expect(textMask == textMaskRealOutput)
        
        let closeEnough = MLMultiArray.allClose(forwardModelOutput, realOutput, rtol: 1e-4, atol: 1e-4)
        #expect(closeEnough)
    }
}

