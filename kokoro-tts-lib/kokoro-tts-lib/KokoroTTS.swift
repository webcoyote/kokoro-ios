//
//  kokoro-tts-lib
//
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
        try espeakNGEngine.setLanguage(dialect(for: voice))
        let normalizedText = normalizeText ? TextNormalizer.normalizeText(text) : text
        let punctuator = Punctuation()
        let (chunks, marks) = punctuator.preserve(text: normalizedText)
        let phonemizedChunks = try chunks.map { try espeakNGEngine.phonemize(text: $0) }
        return Punctuation.restore(text: phonemizedChunks, marks: marks, sep: Separator(word: " "), strip: false).joined(separator: "")
    }
    
    public func generate(text: String, voice: VoiceName) {
        
    }
}
