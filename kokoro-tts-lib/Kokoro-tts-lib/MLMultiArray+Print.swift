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
}
