//
//  kokoro-tts-lib
//
import CoreML

class ProsodyPredictorEngine {
    enum ProsodyPredictionError: Error {
        case modelLoadingFailed
    }
    
    let textEncoder: predictor_text_encoder
    let lstm: predictor_lstm

    init() throws {
        guard let textEncoder = try? predictor_text_encoder(configuration: MLModelConfiguration()) else {
            throw ProsodyPredictionError.modelLoadingFailed
        }
        
        guard let lstm = try? predictor_lstm(configuration: MLModelConfiguration()) else {
            throw ProsodyPredictionError.modelLoadingFailed
        }

        self.textEncoder = textEncoder
        self.lstm = lstm
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
}
