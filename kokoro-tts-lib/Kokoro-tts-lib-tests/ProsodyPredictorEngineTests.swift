//
//  kokoro-tts-lib
//
import Testing
@testable import KokoroTTSLib
import CoreML
import Foundation

class ProsodyPredictorEngineTestsBundleCapturer {
}

struct ProsodyPredictorEngineTests {
    let testBundle = Bundle(for: ProsodyPredictorEngineTestsBundleCapturer.self)
    let prosodyPredictor: ProsodyPredictorEngine
    
    init() throws {
        prosodyPredictor = try ProsodyPredictorEngine()
    }
    
    @Test func testTextEncoder() throws {
        let kokoroTTS = try! KokoroTTS()
        let realInput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "d_en", shape: [1, 512, 143])!
        let realTextMask = try! MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "text_mask_plain", shape: [1, 143])!
        let inputLength = 141
        let referenceVoice = try! kokoroTTS.voiceData(for: .af)
        
        let output = try prosodyPredictor.executeTextEncoder(
            input: realInput,
            referenceVoice: referenceVoice,
            inputLength: Int32(inputLength),
            textMask: realTextMask)
        
        let realOutput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "d", shape: [1, 143, 640])!
        
        let closeEnough = MLMultiArray.allClose(output, realOutput, rtol: 1e-4, atol: 1e-4)
        #expect(closeEnough)
    }
    
    @Test func testLSTM() throws {
        let realInput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "d", shape: [1, 143, 640])!
        let output = try prosodyPredictor.executeLSTM(input: realInput)
        let realOutput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "x", shape: [1, 143, 512])!
        let closeEnough = MLMultiArray.allClose(output, realOutput, rtol: 1e-4, atol: 1e-4)
        #expect(closeEnough)
    }
    
    @Test func testDurationProjection() throws {
        let realInput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "x", shape: [1, 143, 512])!
        let output = try prosodyPredictor.executeDurationProj(input: realInput)
        let realOutput = try MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "duration_input", shape: [1, 143, 50])!
        let closeEnough = MLMultiArray.allClose(output, realOutput, rtol: 1e-4, atol: 1e-4)
        #expect(closeEnough)
    }

    @Test func testCalculateDuration() throws {
        let realInput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "duration_input", shape: [1, 143, 50])!
        let output1 = try prosodyPredictor.calculateDuration(input: realInput)
        let (output, _) = try prosodyPredictor.roundAndClamp(input: output1)
        let realOutput = try! MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "round_and_clamp_output", shape: [1, 143])!
        let closeEnough = MLMultiArray.allClose(output, realOutput, rtol: 1e-8, atol: 1e-8)
        #expect(closeEnough)
    }
    
    @Test func testProcessAlignment() throws {
        let predDur = try! MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "round_and_clamp_output", shape: [1, 143])!
        let d = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "d", shape: [1, 143, 640])!
        let predDurSum = 347

        let output = try prosodyPredictor.processAlignment(
            inputLengths: 143,
            predDurSum: predDurSum,
            predDur: predDur,
            textEncoderOutput: d)
        
        let realOutput = try! MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "en", shape: [1, 640, 347])!
        let closeEnough = MLMultiArray.allClose(output, realOutput, rtol: 1e-3, atol: 1e-3)
        #expect(closeEnough)
    }
}
