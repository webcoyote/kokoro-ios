//
//  kokoro-tts-lib
//
import Foundation
import ESpeakNG

public class KokoroTTS {
    private let espeakNGEngine: ESpeakNGEngine
    
    public enum VoiceName: String {
        case af
        case afBella
        case afSarah
        case amAdam
        case amMichael
        case bfEmma
        case bfIsabella
        case bmGeorge
        case bmLewis
        case afNicole
        case afSky
    }
        
    public init() throws {
        espeakNGEngine = try ESpeakNGEngine()
    }
    
    private func dialect(for voice: VoiceName) -> LanguageDialect {
        switch voice {
            case .af, .afBella, .afSarah, .amAdam, .amMichael, .afNicole, .afSky:
                return .enUS
            case .bfEmma, .bfIsabella, .bmGeorge, .bmLewis:
                return .enGB
        }
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
    
    public func generate(text: String, voice: VoiceName) {
        
    }
}
