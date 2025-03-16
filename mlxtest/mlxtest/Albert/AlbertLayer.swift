//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertLayer {
    let attention: AlbertSelfAttention
    let fullLayerLayerNorm: LayerNorm
    let ffn: Linear
    let ffnOutput: Linear
    let seqLenDim: Int
    
    init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) {
        self.attention = AlbertSelfAttention(weights: weights, config: config, layerNum: layerNum, innerGroupNum: innerGroupNum)
        self.ffn = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.weight"]!,
                          bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.bias"]!)
        self.ffnOutput = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.weight"]!,
                                bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.bias"]!)
        self.seqLenDim = 1
        self.fullLayerLayerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        
        let fullLayerLayerNormWeights = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.weight"]!
        let fullLayerLayerNormBiases = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.bias"]!
        
        guard fullLayerLayerNormWeights.count == config.hiddenSize, fullLayerLayerNormBiases.count == config.hiddenSize else {
            fatalError("Wrong shape for AlbertLayer FullLayerLayerNorm bias or weights!")
        }
        
        for i in 0..<config.hiddenSize {
            self.fullLayerLayerNorm.weight![i] = fullLayerLayerNormWeights[i]
            self.fullLayerLayerNorm.bias![i] = fullLayerLayerNormBiases[i]
        }
    }
    
    func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
        var ffnOutputArray = ffn(attentionOutput)
        ffnOutputArray = MLXNN.gelu(ffnOutputArray)
        ffnOutputArray = ffnOutput(ffnOutputArray)
        return ffnOutputArray
    }
    
    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
        let ffnOutput = ffChunk(attentionOutput)
        let output = fullLayerLayerNorm(ffnOutput + attentionOutput)
        return output
    }
}
