//
//  kokoro-tts-lib
//
import Foundation
import ESpeakNG

class ESpeakNGEngine {
    private var languageSet = false
    private var languageMapping : [String: String] = [:]
    
    enum ESpeakNGEngineError : Error {
        case dataBundleNotFound
        case couldNotInitialize
        case languageNotFound
        case internalError
        case languageNotSet
        case couldNotPhonemize
    }
    
    init() throws {
        if let bundleURLStr = findDataBundlePath() {
            let initOK = espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, bundleURLStr, 0)
            
            if initOK != Constants.successAudioSampleRate {
                log("Internal espseak-ng error, could not initialize")
                throw ESpeakNGEngineError.couldNotInitialize
            }
            
            var languageList: Set<String> = []
            let voiceList = espeak_ListVoices(nil)
            var index = 0
            while let voicePointer = voiceList?.advanced(by: index).pointee {
                let voice = voicePointer.pointee
                if let cLang = voice.languages {
                    let language = String(cString: cLang, encoding: .utf8)!
                        .replacingOccurrences(of: "\u{05}", with: "")
                        .replacingOccurrences(of: "\u{02}", with: "")
                    languageList.insert(language)
                    
                    if let cName = voice.identifier {
                        let name = String(cString: cName, encoding: .utf8)!
                            .replacingOccurrences(of: "\u{05}", with: "")
                            .replacingOccurrences(of: "\u{02}", with: "")
                        languageMapping[language] = name
                    }
                }
                
                index += 1
            }
                        
            try LanguageDialect.allCases.forEach {
                if !languageList.contains($0.rawValue) {
                    log("Language dialect \($0) not found in espeak-ng voice list")
                    throw ESpeakNGEngineError.languageNotFound
                }
            }
            
        } else {
            log("Couldn't find the espeak-ng data bundle, cannot initialize")
            throw ESpeakNGEngineError.dataBundleNotFound
        }
    }
    
    deinit {
        let terminateOK = espeak_Terminate()
        log("ESpeakNGEngine termination OK: \(terminateOK == EE_OK)")
    }
    
    func setLanguage(_ language: LanguageDialect) throws {
        guard let name = languageMapping[language.rawValue] else {
            throw ESpeakNGEngineError.languageNotFound
        }
        
        let result = espeak_SetVoiceByName((name as NSString).utf8String)
        
        if result == EE_NOT_FOUND {
            throw ESpeakNGEngineError.languageNotFound
        } else if result != EE_OK {
            throw ESpeakNGEngineError.internalError
        }
        
        languageSet = true
    }
    
    func phonemize(text: String) throws -> String {
        guard languageSet else {
            throw ESpeakNGEngineError.languageNotSet
        }
        
        guard !text.isEmpty else {
            return ""
        }
        
        var textPtr = UnsafeRawPointer((text as NSString).utf8String)
        let phonemes_mode: Int32 = Int32((Int32(Character("_").asciiValue!) << 8) | 0x02)
                
        let result = withUnsafeMutablePointer(to: &textPtr) { ptr in
            var resultWords: [String] = []
            while ptr.pointee != nil {
                let result = ESpeakNG.espeak_TextToPhonemes(ptr, espeakCHARS_UTF8, phonemes_mode)
                if let result {
                    resultWords.append(String(cString: result, encoding: .utf8)!)
                }
            }
            return resultWords
        }
        
        if !result.isEmpty {
            return result.joined(separator: " ")
        } else {
            throw ESpeakNGEngineError.couldNotPhonemize
        }
    }
    
    private func findDataBundlePath() -> String? {
        if let frameworkBundle = Bundle(identifier: "com.kokoro.espeakng"),
           let dataBundleURL = frameworkBundle.url(forResource: "espeak-ng-data", withExtension: "bundle") {
            return dataBundleURL.path
        }
        return nil
    }
    
    struct Constants {
        static let successAudioSampleRate = 22050
    }
}

