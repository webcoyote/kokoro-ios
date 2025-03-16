//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXFFT

// Hanning window implementation to replace np.hanning
func hanning(length: Int) -> MLXArray {
    if length == 1 {
        return MLXArray(1.0)
    }
    
    let n = MLXArray(Array(stride(from: Float(1-length), to: Float(length), by: 2.0)))
    let factor = .pi / Float(length - 1)
    return 0.5 + 0.5 * cos(n * factor)
}

// Unwrap implementation to replace np.unwrap
func unwrap(p: MLXArray) -> MLXArray {
    let period: Float = 2.0 * .pi
    let discont: Float = period / 2.0
    
    let pDiff1 = p[0..., 0..<p.shape[1] - 1]
    let pDiff2 = p[0..., 1..<p.shape[1]]
    
    let pDiff = pDiff2 - pDiff1
    
    let intervalHigh: Float = period / 2.0
    let intervalLow: Float = -intervalHigh
    
    var pDiffMod = pDiff - intervalLow
    pDiffMod = (((pDiffMod % period) + period) % period) + intervalLow
  
    let ddSignArray = MLX.where(pDiff .> 0, intervalHigh, pDiffMod)
    
    pDiffMod = MLX.where(pDiffMod .== intervalLow, ddSignArray, pDiffMod)
    
    var phCorrect = pDiffMod - pDiff
    phCorrect = MLX.where(abs(pDiff) .< discont, MLXArray(0.0), phCorrect)
            
    return MLX.concatenated([p[0..., 0..<1], (p[0..., 1...] + phCorrect.cumsum(axis: 1))], axis: 1)
}

func mlxStft(
    x: MLXArray,
    nFft: Int = 800,
    hopLength: Int? = nil,
    winLength: Int? = nil,
    window: Any = "hann",
    center: Bool = true,
    padMode: String = "reflect"
) -> MLXArray {
    let hopLen = hopLength ?? nFft / 4
    let winLen = winLength ?? nFft
    
    var w: MLXArray
    if let windowStr = window as? String {
        if windowStr.lowercased() == "hann" {
            w = hanning(length: winLen + 1)[0..<winLen]
        } else {
            fatalError("Only hanning is supported for window, not \(windowStr)")
        }
    } else if let windowArray = window as? MLXArray {
        w = windowArray
    } else {
        fatalError("Window must be a string or MLXArray")
    }
    
    if w.shape[0] < nFft {
        let padSize = nFft - w.shape[0]
        w = MLX.concatenated([w, MLXArray.zeros([padSize])], axis: 0)
    }
    
    func pad(_ x: MLXArray, padding: Int, padMode: String = "reflect") -> MLXArray {
        if padMode == "constant" {
            return MLX.padded(x, width: [padding, padding])
        } else if padMode == "reflect" {
            let prefix = x[1..<padding + 1][.stride(by: -1)]
            let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
            return MLX.concatenated([prefix, x, suffix])
        } else {
            fatalError("Invalid pad mode \(padMode)")
        }
    }
    
    var xArray = x
    
    if center {
        xArray = pad(xArray, padding: nFft / 2, padMode: padMode)
    }
    
    let numFrames = 1 + (xArray.shape[0] - nFft) / hopLen
    if numFrames <= 0 {
        fatalError("Input is too short")
    }
    
    let shape: [Int] = [numFrames, nFft]
    let strides: [Int] = [hopLen, 1]
    
    let frames = MLX.asStrided(xArray, shape, strides: strides)
    
    let spec = MLXFFT.rfft(frames * w)
    return spec.transposed(1, 0)
}

func mlxIstft(
    x: MLXArray,
    hopLength: Int? = nil,
    winLength: Int? = nil,
    window: Any = "hann",
    center: Bool = true,
    length: Int? = nil
) -> MLXArray {
    let winLen = winLength ?? (x.shape[1] - 1) * 2
    let hopLen = hopLength ?? winLen / 4
    
    var w: MLXArray
    if let windowStr = window as? String {
        if windowStr.lowercased() == "hann" {
            w = hanning(length: winLen + 1)[0..<winLen]
        } else {
            fatalError("Only hanning window is supported")
        }
    } else if let windowArray = window as? MLXArray {
        w = windowArray
    } else {
        fatalError("Window must be a string or MLXArray")
    }
    
    if w.shape[0] < winLen {
        w = MLX.concatenated([w, MLXArray.zeros([winLen - w.shape[0]])], axis: 0)
    }
    
    let xTransposed = x.transposed(1, 0)
    let t = (xTransposed.shape[0] - 1) * hopLen + winLen
    var reconstructed = MLXArray.zeros([t])
    let windowSum = MLXArray.zeros([t])
    
    for i in 0..<xTransposed.shape[0] {
        // Inverse FFT of each frame
        let frameTime = MLXFFT.irfft(xTransposed[i])
        
        // Get the position in the time-domain signal to add the frame
        let start = i * hopLen
        let end = start + winLen
        
        // overlap-add the inverse transformed frame, scaled by the window
        reconstructed[start..<end] = reconstructed[start..<end] + frameTime * w
        windowSum[start..<end] = windowSum[start..<end] + w * w
    }
    
    // Normalize by the sum of the window values
    reconstructed = MLX.where(windowSum .!= 0, reconstructed / windowSum, reconstructed)

    if center && length == nil {
        reconstructed = reconstructed[winLen / 2..<(reconstructed.shape[0] - winLen / 2)]
    }
    
    if let length = length {
        reconstructed = reconstructed[0..<length]
    }
    
    return reconstructed
}

class MLXSTFT {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int
    let window: String
    
    var magnitude: MLXArray?
    var phase: MLXArray?
    
    init(filterLength: Int = 800, hopLength: Int = 200, winLength: Int = 800, window: String = "hann") {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength
        self.window = window
    }
    
    func transform(inputData: MLXArray) -> (MLXArray, MLXArray) {
        var audioArray = inputData
        if audioArray.ndim == 1 {
            audioArray = audioArray.expandedDimensions(axis: 0)
        }
        
        var magnitudes: [MLXArray] = []
        var phases: [MLXArray] = []
        
        for batchIdx in 0..<audioArray.shape[0] {
            // Compute STFT
            let stft = mlxStft(
                x: audioArray[batchIdx],
                nFft: filterLength,
                hopLength: hopLength,
                winLength: winLength,
                window: window,
                center: true,
                padMode: "reflect"
            )
            
            let magnitude = MLX.abs(stft)
            
            // Replaces np.angle()
            let phase = MLX.atan2(stft.imaginaryPart(), stft.realPart())
            
            magnitudes.append(magnitude)
            phases.append(phase)
        }
        
        let magnitudesStacked = MLX.stacked(magnitudes, axis: 0)
        let phasesStacked = MLX.stacked(phases, axis: 0)
        
        return (magnitudesStacked, phasesStacked)
    }
    
    func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        var reconstructed: [MLXArray] = []
        
        for batchIdx in 0..<magnitude.shape[0] {
            let phaseCont = unwrap(p: phase[batchIdx])
                        
            // Combine magnitude and phase
            let stft = magnitude[batchIdx] * MLX.exp(MLXArray(real: 0, imaginary: 1) * phaseCont)
                        
            // Inverse STFT
            let audio = mlxIstft(
                x: stft,
                hopLength: hopLength,
                winLength: winLength,
                window: window,
                center: true,
                length: nil
            )
            
            reconstructed.append(audio)
        }
        
        let reconstructedStacked = MLX.stacked(reconstructed, axis: 0)
        return reconstructedStacked.expandedDimensions(axis: 1)
    }
    
    func callAsFunction(inputData: MLXArray) -> MLXArray {
        let (mag, ph) = transform(inputData: inputData)
        self.magnitude = mag
        self.phase = ph
        let reconstruction = inverse(magnitude: mag, phase: ph)
        return reconstruction.expandedDimensions(axis: -2)
    }
} 
