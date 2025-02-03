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
    
    private func phonemize(text: String, lang: String, normalizeText: Bool = false) -> String {
        var phonemizedText = normalizeText ? TextNormalizer.normalizeText(text) : text
        
        return phonemizedText
    }
    
    public func generate(text: String, voice: VoiceName) {
        
    }
}
