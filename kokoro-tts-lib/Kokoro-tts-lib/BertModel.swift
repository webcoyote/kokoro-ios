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
    
    /// Given an array of lengths, returns an MLMultiArray mask of shape [numLengths, maxLength]
    /// where for each row the first `length` elements are 1 (valid) and the rest are 0.
    func lengthToMask(lengths: [Int]) throws -> MLMultiArray {
        // Get the maximum length (will be the number of columns)
        guard let maxLength = lengths.max() else {
            throw BertModelError.lengthToMaskError
        }
        let numLengths = lengths.count
        
        // Create an MLMultiArray with shape [numLengths, maxLength] of type Int32
        let maskShape: [NSNumber] = [NSNumber(value: numLengths), NSNumber(value: maxLength)]
        let maskArray = try MLMultiArray(shape: maskShape, dataType: .int32)
        
        // For each row, set positions less than the length to 1 (valid) and the rest to 0.
        for i in 0..<numLengths {
            let currentLength = lengths[i]
            for j in 0..<maxLength {
                // This logic mirrors the PyTorch code:
                // In PyTorch: mask = (torch.arange(max) + 1 > length) then inverted (~mask).int()
                // which is equivalent to: positions j where j < length become 1.
                let value: Int32 = (j < currentLength) ? 1 : 0
                maskArray[[NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: value)
            }
        }
        return maskArray
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
        
        return (output.sequence_output, tokensArray, textMask, tokensCount)
    }
    
    func validateShape(of mlMultiArray: MLMultiArray) -> Bool {
        let shape = mlMultiArray.shape.map { $0.intValue }
        return shape.count == 3 && shape[0] == 1 && (1...1024).contains(shape[1]) && shape[2] == 768
    }
    
    private func transposeLastTwoDimensions(of mlArray: MLMultiArray) throws -> MLMultiArray {
        let shape = mlArray.shape.map { $0.intValue }
        let batch = shape[0]
        let rows = shape[1]
        let cols = shape[2]
                
        let newMultiArrayShape = [batch, cols, rows]
        let transposedArray = try MLMultiArray(shape: newMultiArrayShape.map { NSNumber(value: $0) }, dataType: mlArray.dataType)
            
        for b in 0..<batch {
            for i in 0..<rows {
                for j in 0..<cols {
                    let oldIndex = [NSNumber(value: b), NSNumber(value: i), NSNumber(value: j)]
                    let newIndex = [NSNumber(value: b), NSNumber(value: j), NSNumber(value: i)]
                    transposedArray[newIndex] = mlArray[oldIndex]
                }
            }
        }
        return transposedArray
    }
    
    func forwardPassOnEncoderModel(array: MLMultiArray) throws -> MLMultiArray {
        guard validateShape(of: array) else {
            throw BertModelError.invalidInputArgument
        }
        
        let input = bert_encoderInput(input: array)
        let output = try encoderModel.prediction(input: input)
        let transposedOutput = try transposeLastTwoDimensions(of: output.var_4)
                
        return transposedOutput
    }
}
