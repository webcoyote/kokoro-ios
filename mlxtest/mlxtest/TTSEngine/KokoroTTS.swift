//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Available voices
public enum TTSVoice {
  case afHeart
  case bmGeorge
}

// Main class, encapsulates the whole Kokoro text-to-speech pipeline
public class KokoroTTS {
  enum KokoroTTSError: Error {
    case tooManyTokens
  }

  private let bert: CustomAlbert!
  private let bertEncoder: Linear!
  private let durationEncoder: DurationEncoder!
  private let predictorLSTM: LSTM!
  private let durationProj: Linear!
  private let prosodyPredictor: ProsodyPredictor!
  private let textEncoder: TextEncoder!
  private let decoder: Decoder!
  private let eSpeakEngine: ESpeakNGEngine!
  private var chosenVoice: TTSVoice?
  private var voice: MLXArray!

  init() {
    let sanitizedWeights = WeightLoader.loadWeights()

    bert = CustomAlbert(weights: sanitizedWeights, config: AlbertModelArgs())
    bertEncoder = Linear(weight: sanitizedWeights["bert_encoder.weight"]!, bias: sanitizedWeights["bert_encoder.bias"]!)
    durationEncoder = DurationEncoder(weights: sanitizedWeights, dModel: 512, styDim: 128, nlayers: 6)

    predictorLSTM = LSTM(
      inputSize: 512 + 128,
      hiddenSize: 512 / 2,
      wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
      whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
      biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
      biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
      wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
      whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
    )

    durationProj = Linear(
      weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
      bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
    )

    prosodyPredictor = ProsodyPredictor(
      weights: sanitizedWeights,
      styleDim: 128,
      dHid: 512
    )

    textEncoder = TextEncoder(
      weights: sanitizedWeights,
      channels: 512,
      kernelSize: 5,
      depth: 3,
      nSymbols: 178
    )

    decoder = Decoder(
      weights: sanitizedWeights,
      dimIn: 512,
      styleDim: 128,
      dimOut: 80,
      resblockKernelSizes: [3, 7, 11],
      upsampleRates: [10, 6],
      upsampleInitialChannel: 512,
      resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
      upsampleKernelSizes: [20, 12],
      genIstftNFft: 20,
      genIstftHopSize: 5
    )

    eSpeakEngine = try! ESpeakNGEngine()
  }

  public func generateAudio(voice: TTSVoice, text: String, speed: Float = 1.0) throws -> MLXArray {
    BenchmarkTimer.shared.create(id: "Phonemization", parent: "TTSGeneration")

    if chosenVoice != voice {
      self.voice = VoiceLoader.loadVoice(voice)
      try eSpeakEngine.setLanguage(for: voice)
      chosenVoice = voice
    }

    let outputStr = try! eSpeakEngine.phonemize(text: text)

    let inputIds = Tokenizer.tokenize(phonemizedText: outputStr)
    guard inputIds.count <= Constants.maxTokenCount else {
      throw KokoroTTSError.tooManyTokens
    }

    let paddedInputIdsBase = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])

    let inputLengths = MLXArray(paddedInputIds.dim(-1))
    let inputLengthMax: Int = inputLengths.max().item()
    var textMask = MLXArray(0 ..< inputLengthMax)
    textMask = textMask + 1 .> inputLengths
    textMask = textMask.expandedDimensions(axes: [0])
    let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
    let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
    let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)

    let (bertDur, _) = bert(paddedInputIds, attentionMask: attentionMask)
    let dEn = bertEncoder(bertDur).transposed(0, 2, 1)

    BenchmarkTimer.shared.stop(id: "Phonemization")
    BenchmarkTimer.shared.create(id: "Duration", parent: "TTSGeneration")

    let refS = self.voice[inputIds.count - 1, 0 ... 1, 0...]
    let s = refS[0 ... 1, 128...]

    let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
    let (x, _) = predictorLSTM(d)
    let duration = durationProj(x)
    let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
    let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]

    let indices = MLX.concatenated(
      predDur.enumerated().map { i, n in
        let nSize: Int = n.item()
        return MLX.repeated(MLXArray([i]), count: nSize)
      }
    )

    var swiftPredAlnTrg = [Float](repeating: 0.0, count: indices.shape[0] * paddedInputIds.shape[1])
    for i in 0 ..< indices.shape[0] {
      let indiceValue: Int = indices[i].item()
      swiftPredAlnTrg[indiceValue * indices.shape[0] + i] = 1.0
    }
    let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([paddedInputIds.shape[1], indices.shape[0]])
    let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
    let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)

    BenchmarkTimer.shared.stop(id: "Duration")
    BenchmarkTimer.shared.create(id: "Prosody", parent: "TTSGeneration")

    let (F0Pred, NPred) = prosodyPredictor.F0NTrain(x: en, s: s)
    let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    let asr = MLX.matmul(tEn, predAlnTrg)

    BenchmarkTimer.shared.stop(id: "Prosody")
    BenchmarkTimer.shared.create(id: "Decoder", parent: "TTSGeneration")

    let audio = decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: refS[0 ... 1, 0 ... 127])[0]
    audio.eval()

    BenchmarkTimer.shared.stop(id: "Decoder")

    return audio
  }

  struct Constants {
    static let maxTokenCount = 510        
  }
}
