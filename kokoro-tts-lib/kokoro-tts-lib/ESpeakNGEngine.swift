//
//  kokoro-tts-lib
//
import Foundation
import ESpeakNG

class ESpeakNGEngine {
    enum ESpeakNGEngineError : Error {
        case dataBundleNotFound
        case couldNotInitialize
        case languageNotFound
        case internalError
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
                    languageList.insert(String(cString: cLang))
                }
                index += 1
            }
            
            for dialect in LanguageDialect.allCases {
                if !languageList.contains(dialect.rawValue) {
                    log("Language dialect \(dialect) not found in espeak-ng voice list")
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
    
    private func createVoiceStruct(_ language: String) -> espeak_VOICE {
        espeak_VOICE(
            name: nil,
            languages: (language as NSString).utf8String,
            identifier: nil,
            gender: 0,
            age: 0,
            variant: 0,
            xx1: 0,
            score: 0,
            spare: nil
        )
    }
    
    func setLanguage(_ language: LanguageDialect) throws {
        var voiceStruct = createVoiceStruct(language.rawValue)
        
        let result = withUnsafeMutablePointer(to: &voiceStruct) { voicePtr in
            espeak_SetVoiceByProperties(voicePtr)
        }
        
        if result == EE_NOT_FOUND {
            throw ESpeakNGEngineError.languageNotFound
        } else if result != EE_OK {
            throw ESpeakNGEngineError.internalError
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

