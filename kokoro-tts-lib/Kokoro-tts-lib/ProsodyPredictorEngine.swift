//
//  kokoro-tts-lib
//
import CoreML
internal import MLX

class ProsodyPredictorEngine {
    enum ProsodyPredictionError: Error {
        case modelLoadingFailed
        case couldNotCreateArray
        case wrongInput
    }
    
    let textEncoder: predictor_text_encoder
    let lstm: predictor_lstm
    let durationProj: predictor_duration_proj
    
    init() throws {
        guard let textEncoder = try? predictor_text_encoder(configuration: MLModelConfiguration()) else {
            throw ProsodyPredictionError.modelLoadingFailed
        }
        
        guard let lstm = try? predictor_lstm(configuration: MLModelConfiguration()) else {
            throw ProsodyPredictionError.modelLoadingFailed
        }
        
        guard let durationProj = try? predictor_duration_proj(configuration: MLModelConfiguration()) else {
            throw ProsodyPredictionError.modelLoadingFailed
        }
        
        self.textEncoder = textEncoder
        self.lstm = lstm
        self.durationProj = durationProj
    }
    
    func executeTextEncoder(input: MLMultiArray, referenceVoice: MLMultiArray, inputLength: Int32, textMask: MLMultiArray) throws -> MLMultiArray {
        let inputLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        let inputLengthIndex = [NSNumber(value: 0)]
        inputLengthArray[inputLengthIndex] = inputLength as NSNumber
        
        let referenceVoiceLength = referenceVoice.shape[2].intValue - 128
        
        let cutReferenceVoiceArray = try MLMultiArray(shape: [1, referenceVoiceLength as NSNumber], dataType: .float32)
        for j in 128..<referenceVoice.shape[2].intValue {
            let index = [NSNumber(value: 0), NSNumber(value: j - 128)]
            let val = referenceVoice[[inputLength as NSNumber, 0, j as NSNumber]] as NSNumber
            cutReferenceVoiceArray[index] = val
        }
        
        let inputs = predictor_text_encoderInput(x_1: input, style: cutReferenceVoiceArray, text_lengths: inputLengthArray, m_1: textMask)
        let outputs = try textEncoder.prediction(input: inputs)
        
        return outputs.var_462
    }
    
    func executeLSTM(input: MLMultiArray) throws -> MLMultiArray {
        let inputs = predictor_lstmInput(input: input)
        let outputs = try lstm.prediction(input: inputs)
        return outputs.var_39
    }
    
    func executeDurationProj(input: MLMultiArray) throws -> MLMultiArray {
        let outputs = try durationProj.prediction(input: predictor_duration_projInput(x: input))
        return outputs.var_5
    }
    
    func calculateDuration(input: MLMultiArray) throws -> MLMultiArray {
        func sigmoid(_ x: Double) -> Double {
            return 1 / (1 + exp(-x))
        }
        
        let resultShape = [1, input.shape[1]] as [NSNumber]
        let result = try MLMultiArray(shape: resultShape, dataType: .float32)
        
        for i in 0..<input.shape[1].intValue {
            var sum: Double = 0.0
            for f in 0..<input.shape[2].intValue {
                let value = input[[0, i as NSNumber, f as NSNumber]].doubleValue
                sum += sigmoid(value)
            }
            result[[0, i as NSNumber]] = NSNumber(value: sum)
        }
        
        return result
    }
    
    func roundAndClamp(input: MLMultiArray) throws -> (MLMultiArray, Int) {
        let resultShape = input.shape
        guard let result = try? MLMultiArray(shape: resultShape, dataType: .float32) else {
            throw ProsodyPredictionError.couldNotCreateArray
        }
            
        var durationSum = 0
        for i in 0..<input.count {
            let roundedValue = round(input[i].doubleValue)
            let clampedValue = Int(max(roundedValue, 1.0))
            result[i] = NSNumber(value: clampedValue)
            durationSum += clampedValue
        }
            
        return (result, durationSum)
    }
    
    func processAlignment(inputLengths: Int, predDurSum: Int, predDur: MLMultiArray, textEncoderOutput: MLMultiArray) throws -> MLMultiArray {
        let predAlnTrgShape = [inputLengths, predDurSum] as [NSNumber]
        let predAlnTrg = try MLMultiArray(shape: predAlnTrgShape, dataType: .float32)
        for i in 0..<predAlnTrg.count { predAlnTrg[i] = 0 }
        
        var cFrame = 0
        for i in 0..<inputLengths {
            let duration = predDur[i].intValue
            for j in cFrame..<cFrame + duration {
                predAlnTrg[i * predDurSum + j] = 1
            }
            cFrame += duration
        }
        
        print("ASDFADSFDSA ", predAlnTrg.shape)
                
        guard let dTransposed = try? textEncoderOutput.transposeLastTwoDimensions(),
              let predAlnTrgExpanded = predAlnTrg.expandDimsFrom2Dto3D() else {
            throw ProsodyPredictionError.couldNotCreateArray
        }
        
        print("DRTR ", dTransposed.shape)
        print("ORED ", predAlnTrgExpanded.shape)
        
        guard let en = batchedMatMul(dTransposed, predAlnTrgExpanded) else {
            throw ProsodyPredictionError.couldNotCreateArray
        }
        
        return en
    }
}
