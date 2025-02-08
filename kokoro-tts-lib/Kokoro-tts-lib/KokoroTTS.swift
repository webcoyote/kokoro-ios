//
//  kokoro-tts-lib
//
import Foundation
import ESpeakNG
import CoreML

public class KokoroTTS {
    enum KokoroTTSError: Error {
        case failedToLoadVoiceModel
    }
    let espeakNGEngine: ESpeakNGEngine
    let bertModel: BertModel
    let prosodyPredictorEngine: ProsodyPredictorEngine
    
    var voiceDataCache: [VoiceName: MLMultiArray] = [:]
    
    public enum VoiceName: String {
        case af
        // case afBella
        // case afSarah
        // case amAdam
        // case amMichael
        // case bfEmma
        // case bfIsabella
        // case bmGeorge
        // case bmLewis
        // case afNicole
        // case afSky
    }
        
    private func loadVoicePackJSON(fileName: String) throws -> MLMultiArray {
        let kokoroBundle = Bundle(for: KokoroTTS.self)
        guard let path = kokoroBundle.path(forResource: fileName, ofType: "json") else {
            throw KokoroTTSError.failedToLoadVoiceModel
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
        let mlArray = try MLMultiArray(shape: [511, 1, 256], dataType: .float32)
        
        if let nestedArray = jsonObject as? [[[Any]]] {
            guard nestedArray.count == 511 else { throw KokoroTTSError.failedToLoadVoiceModel }
            for a in 0..<nestedArray.count {
                guard nestedArray[a].count == 1 else { throw KokoroTTSError.failedToLoadVoiceModel }
                for b in 0..<nestedArray[a].count {
                    guard nestedArray[a][b].count == 256 else { throw KokoroTTSError.failedToLoadVoiceModel }
                    for c in 0..<nestedArray[a][b].count {
                        mlArray[[a as NSNumber, b as NSNumber, c as NSNumber]] = nestedArray[a][b][c] as! NSNumber
                    }
                }
            }
        } else {
            throw KokoroTTSError.failedToLoadVoiceModel
        }
        
        return mlArray
    }
    
    public init() throws {
        espeakNGEngine = try ESpeakNGEngine()
        bertModel = try BertModel()
        prosodyPredictorEngine = try ProsodyPredictorEngine()
    }
    
    private func dialect(for voice: VoiceName) -> LanguageDialect {
        switch voice {
        case .af:
            return .enUS            
        }
    }
    
    func voiceData(for voice: VoiceName) throws -> MLMultiArray {
        if voiceDataCache[voice] == nil {
            let bundle = Bundle(for: KokoroTTS.self)
            
            switch voice {
            case .af:
                voiceDataCache[voice] = try MLMultiArray.read3DArrayFromJson(bundle: bundle, file: "voicepack_af_weights", shape: [511, 1, 256]) 
            }
        }
        return voiceDataCache[voice]!
    }
    
    func phonemize(text: String, using voice: VoiceName, normalizeText: Bool = true) throws -> String {
        log("Starting to phonemize text: \(text)")
        let language = dialect(for: voice)
        try espeakNGEngine.setLanguage(language)
        let normalizedText = normalizeText ? TextNormalizer.normalizeText(text) : text
        log("Normalized text: \(normalizedText)")
        let punctuator = Punctuation()
        let (chunks, marks) = punctuator.preserve(text: normalizedText)
        let phonemizedChunks = try chunks.map {
            TextNormalizer.postprocess(try espeakNGEngine.phonemize(text: $0), separator: Separator(word: " "), strip: true)
        }
        log("Phonemized chunks: \(phonemizedChunks)")
        let restoredText = Punctuation.restore(text: phonemizedChunks, marks: marks, sep: Separator(word: " "), strip: false).joined(separator: "")
        let normalizedPhonemizedText = TextNormalizer.postPhonemizeNormalize(text: restoredText, language: language)
        log("Final output: \(normalizedPhonemizedText)")
        return normalizedPhonemizedText
    }
    
    public func generate(text: String, voice: VoiceName) throws {
        let phonemizedText = try phonemize(text: text, using: voice)
        let tokenizedText = TextNormalizer.tokenize(phonemizedText)
        
        let bertModelOutput: MLMultiArray
        let textMask: MLMultiArray
        let totalTokenCount: Int
                
        (bertModelOutput, _, textMask, totalTokenCount) = try bertModel.forwardPassOnInitialModel(tokens: tokenizedText)
        let forwardModelOutput = try bertModel.forwardPassOnEncoderModel(array: bertModelOutput)
                
        let textEncoderOutput = try prosodyPredictorEngine.executeTextEncoder(
            input: forwardModelOutput,
            referenceVoice: try voiceData(for: voice),
            inputLength: Int32(tokenizedText.count),
            textMask: textMask)
        
        let lstmOutput = try prosodyPredictorEngine.executeLSTM(input: textEncoderOutput)
        let predictorOutput = try prosodyPredictorEngine.executeDurationProj(input: lstmOutput)
        let durationProj = try prosodyPredictorEngine.executeDurationProj(input: lstmOutput)
        let duration = try prosodyPredictorEngine.calculateDuration(input: durationProj)
        let roundAndClamp = try prosodyPredictorEngine.roundAndClamp(input: duration)        
    }
}
