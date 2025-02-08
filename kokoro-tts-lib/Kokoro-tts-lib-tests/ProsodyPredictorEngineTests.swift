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
        
        let closeEnough2 = MLMultiArray.allClose(output, realOutput, rtol: 1e-4, atol: 1e-4)
        #expect(closeEnough2)
    }

    @Test func testCalculateDuration() throws {
        guard let durationInput = try MLMultiArray.read3DArrayFromJson(bundle: testBundle, file: "duration_input", shape: [1, 143, 50]),
              let durationRealOutput = try MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "duration_output", shape: [1, 143]),
              let roundAndClampRealOutput = try MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "round_and_clamp_output", shape: [1, 143])
        else {
            Issue.record("Could not load duration input")
            return
        }
        
        
        let duration = try prosodyPredictor.calculateDuration(input: durationInput)
        #expect(duration == durationRealOutput)
        
        let roundAndClampOutput: MLMultiArray
        let durationSum: Int
        
        (roundAndClampOutput, durationSum) = try prosodyPredictor.roundAndClamp(input: duration)
        #expect(roundAndClampOutput == roundAndClampRealOutput)
        
        // Test code
        print("DUR SUM ", durationSum)
        let predAlnTrgShape = [143, durationSum] as [NSNumber]
        let predAlnTrg = try MLMultiArray(shape: predAlnTrgShape, dataType: .float32)
        for i in 0..<predAlnTrg.count { predAlnTrg[i] = 0 }
        
        var cFrame = 0
        for i in 0..<143 {
            let duration = roundAndClampOutput[i].intValue
            for j in cFrame..<cFrame + duration {
                predAlnTrg[i * durationSum + j] = 1
            }
            cFrame += duration
        }
        
        let expectedOutput = try MLMultiArray.read2DArrayFromJson(bundle: testBundle, file: "pred_aln_trg_output", shape: [143, 347])
        
        #expect(predAlnTrg.shape == expectedOutput!.shape)
        
        #expect(predAlnTrg == expectedOutput)
    }
}
