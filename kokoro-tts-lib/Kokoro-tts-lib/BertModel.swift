//
//  kokoro-tts-lib
//
import CoreML

class BertModel {
    let initialModel: bert_model
    let encoderModel: bert_encoder
    
    enum BertModelError: Error {
        case modelLoadingFailed
        case lengthToMaskError
        case invalidInputArgument
    }
    
    init() throws {
        guard let initialModel = try? bert_model(configuration: MLModelConfiguration()) else {
            throw BertModelError.modelLoadingFailed
        }
        
        guard let encoderModel = try? bert_encoder(configuration: MLModelConfiguration()) else {
            throw BertModelError.modelLoadingFailed
        }
                
        self.initialModel = initialModel
        self.encoderModel = encoderModel
    }
    
    func lengthToMask(lengths: [Int]) throws -> MLMultiArray {
        guard let maxLength = lengths.max() else {
            throw BertModelError.lengthToMaskError
        }
        let numLengths = lengths.count
        
        let maskShape: [NSNumber] = [NSNumber(value: numLengths), NSNumber(value: maxLength)]
        let maskArray = try MLMultiArray(shape: maskShape, dataType: .int32)
        
        for i in 0..<numLengths {
            let currentLength = lengths[i]
            for j in 0..<maxLength {
                let value: Int32 = (j < currentLength) ? 1 : 0
                maskArray[[NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: value)
            }
        }
        return maskArray
    }
    
    func lengthToMaskPlain(length: Int) throws -> MLMultiArray {
        let maskShape = [1, length] as [NSNumber]
        let mask = try MLMultiArray(shape: maskShape, dataType: .int32)
        for i in 0..<mask.count { mask[i] = 0 }
        return mask
    }
    
    func forwardPassOnInitialModel(tokens: [Int]) throws -> (MLMultiArray, MLMultiArray, MLMultiArray, Int) {
        let tokensWithPadding = [0] + tokens + [0]
        let tokensCount = tokensWithPadding.count
            
        let tokensShape: [NSNumber] = [1, NSNumber(value: tokensCount)]
        let tokensArray = try MLMultiArray(shape: tokensShape, dataType: .int32)
            
        // Fill the tokens MLMultiArray.
        for i in 0..<tokensCount {
            tokensArray[[0, NSNumber(value: i)]] = NSNumber(value: tokensWithPadding[i])
        }
            
        let inputLengths = [tokensCount]
        let textMask = try lengthToMask(lengths: inputLengths)
                    
        let inputs = bert_modelInput(input_ids: tokensArray, mask: textMask)
        let output: bert_modelOutput = try initialModel.prediction(input: inputs)
        
        return (output.sequence_output, tokensArray, try lengthToMaskPlain(length: tokensCount), tokensCount)
    }
    
    func validateShape(of mlMultiArray: MLMultiArray) -> Bool {
        let shape = mlMultiArray.shape.map { $0.intValue }
        return shape.count == 3 && shape[0] == 1 && (1...1024).contains(shape[1]) && shape[2] == 768
    }
    
    func forwardPassOnEncoderModel(array: MLMultiArray) throws -> MLMultiArray {
        guard validateShape(of: array) else {
            throw BertModelError.invalidInputArgument
        }
        
        let input = bert_encoderInput(input: array)
        let output = try encoderModel.prediction(input: input)
        let transposedOutput = try output.var_4.transposeLastTwoDimensions()
                
        return transposedOutput
    }
}
