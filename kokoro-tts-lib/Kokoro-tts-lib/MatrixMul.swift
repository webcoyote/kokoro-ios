import Foundation
import CoreML

/// Performs batched matrix multiplication on two 3D MLMultiArrays (without using Accelerate).
/// - Parameters:
///   - a: An MLMultiArray with shape [batch, m, k] and data type Float32.
///   - b: An MLMultiArray with shape [batch, k, n] and data type Float32.
/// - Returns: An MLMultiArray with shape [batch, m, n] containing the product, or nil if shapes are incompatible.
func batchedMatMul(_ a: MLMultiArray, _ b: MLMultiArray) -> MLMultiArray? {
    // Verify that both arrays are 3-dimensional.
    guard a.shape.count == 3, b.shape.count == 3 else {
        print("Both arrays must be 3-dimensional.")
        return nil
    }
    
    // Extract dimensions.
    // Array 'a' shape: [batch, m, k]
    let batch = a.shape[0].intValue
    let m = a.shape[1].intValue
    let k = a.shape[2].intValue
    
    // Array 'b' shape: [batch, k, n]
    let batchB = b.shape[0].intValue
    let kB = b.shape[1].intValue
    let n = b.shape[2].intValue
    
    // Ensure that the batch sizes and inner dimensions match.
    guard batch == batchB, k == kB else {
        print("Shape mismatch: a is [\(batch), \(m), \(k)] and b is [\(batchB), \(kB), \(n)].")
        return nil
    }
    
    // Create the output MLMultiArray with shape [batch, m, n].
    guard let result = try? MLMultiArray(shape: [NSNumber(value: batch),
                                                   NSNumber(value: m),
                                                   NSNumber(value: n)],
                                          dataType: .float32) else {
        print("Could not create the result MLMultiArray.")
        return nil
    }
    
    // Loop over each batch.
    for batchIndex in 0..<batch {
        // Loop over rows of the result (from 'a').
        for i in 0..<m {
            // Loop over columns of the result (from 'b').
            for j in 0..<n {
                var sum: Float = 0.0
                // Compute the dot product for the (i, j) element.
                for p in 0..<k {
                    // Build index arrays for accessing elements.
                    let indexA: [NSNumber] = [NSNumber(value: batchIndex),
                                                NSNumber(value: i),
                                                NSNumber(value: p)]
                    let indexB: [NSNumber] = [NSNumber(value: batchIndex),
                                                NSNumber(value: p),
                                                NSNumber(value: j)]
                    
                    // Retrieve the elements and convert to Float.
                    let aValue = a[indexA].floatValue
                    let bValue = b[indexB].floatValue
                    
                    sum += aValue * bValue
                }
                // Store the computed sum in the result array.
                let resultIndex: [NSNumber] = [NSNumber(value: batchIndex),
                                               NSNumber(value: i),
                                               NSNumber(value: j)]
                result[resultIndex] = NSNumber(value: sum)
            }
        }
    }
    
    return result
}
