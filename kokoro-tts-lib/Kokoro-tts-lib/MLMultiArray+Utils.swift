//
//  kokoro-tts-lib
//
import CoreML
import Foundation

extension MLMultiArray {
    func print() {
        guard self.count > 0 else {
            log("MLMultiArray is empty.")
            return
        }
            
        let pointer = UnsafeMutablePointer<Float32>(OpaquePointer(self.dataPointer))
        let buffer = UnsafeBufferPointer(start: pointer, count: self.count)
        let array = Array(buffer)
                    
        Swift.print(array)
    }
    
    func transposeLastTwoDimensions() throws -> MLMultiArray {
        guard self.shape.count == 3 else { throw NSError(domain: "MLMultiArrayExtensionError", code: -1, userInfo: nil)}
        
        let shape = self.shape.map { $0.intValue }
        let batch = shape[0]
        let rows = shape[1]
        let cols = shape[2]
                
        let newMultiArrayShape = [batch, cols, rows]
        let transposedArray = try MLMultiArray(shape: newMultiArrayShape.map { NSNumber(value: $0) }, dataType: self.dataType)
            
        for b in 0..<batch {
            for i in 0..<rows {
                for j in 0..<cols {
                    let oldIndex = [NSNumber(value: b), NSNumber(value: i), NSNumber(value: j)]
                    let newIndex = [NSNumber(value: b), NSNumber(value: j), NSNumber(value: i)]
                    transposedArray[newIndex] = self[oldIndex]
                }
            }
        }
        return transposedArray
    }
    
    static func read3DArrayFromJson(bundle: Bundle, file: String, shape: [Int], dataType: MLMultiArrayDataType = .float32) throws -> MLMultiArray? {
        guard shape.count == 3 else { return nil }
        
        guard let path = bundle.path(forResource: file, ofType: "json") else {
            return nil
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
        let array = try MLMultiArray(shape: [shape[0] as NSNumber, shape[1] as NSNumber, shape[2] as NSNumber], dataType: dataType)
        
        if let nestedArray = jsonObject as? [[[Any]]] {
            guard nestedArray.count == shape[0] else { return nil }
            for a in 0..<nestedArray.count {
                guard nestedArray[a].count == shape[1] else { return nil }
                for b in 0..<nestedArray[a].count {
                    guard nestedArray[a][b].count == shape[2] else { return nil }
                    for c in 0..<nestedArray[a][b].count {
                        array[[a as NSNumber, b as NSNumber, c as NSNumber]] = nestedArray[a][b][c] as! NSNumber
                    }
                }
            }
        } else {
            return nil
        }
        
        return array
    }
    
    static func read2DArrayFromJson(bundle: Bundle, file: String, shape: [Int], dataType: MLMultiArrayDataType = .float32) throws -> MLMultiArray? {
        guard shape.count == 2 else { return nil }
        
        guard let path = bundle.path(forResource: file, ofType: "json") else {
            return nil
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
        let array = try MLMultiArray(shape: [shape[0] as NSNumber, shape[1] as NSNumber], dataType: dataType)
        
        if let nestedArray = jsonObject as? [[Any]] {
            guard nestedArray.count == shape[0] else { return nil }
            for a in 0..<nestedArray.count {
                guard nestedArray[a].count == shape[1] else { return nil }
                for b in 0..<nestedArray[a].count {
                    array[[a as NSNumber, b as NSNumber]] = nestedArray[a][b] as! NSNumber
                }
            }
        } else {
            return nil
        }
        
        return array
    }
    
    static func allClose(_ lhs: MLMultiArray, _ rhs: MLMultiArray, rtol: Double = 1e-5, atol: Double = 1e-8) -> Bool {
        if lhs.shape != rhs.shape || lhs.count != rhs.count {
            return false
        }
        
        for i in 0..<lhs.count {
            let lhsValue = lhs[i].doubleValue
            let rhsValue = rhs[i].doubleValue
            
            let tolerance = atol + rtol * abs(rhsValue)
            
            if abs(lhsValue - rhsValue) > tolerance {
                return false
            }
        }
        
        return true
    }
    
    func expandDimsFrom2Dto3D() -> MLMultiArray? {
        guard shape.count == 2 else {
            return nil
        }
        
        let dim0 = shape[0].intValue
        let dim1 = shape[1].intValue
        
        guard let output = try? MLMultiArray(shape: [1, NSNumber(value: dim0), NSNumber(value: dim1)], dataType: dataType) else {
            return nil
        }
        
        for i in 0..<dim0 {
            for j in 0..<dim1 {
                let inputIndex: [NSNumber] = [NSNumber(value: i), NSNumber(value: j)]
                let outputIndex: [NSNumber] = [0, NSNumber(value: i), NSNumber(value: j)]
                output[outputIndex] = self[inputIndex]
            }
        }
        
        return output
    }
}
