import Foundation
import ESpeakNG

class ESpeakNGTest {
    private init() {}
    
    static private func findDataBundlePath() -> String? {
        if let frameworkBundle = Bundle(identifier: "com.kokoro.espeakng"),
           let dataBundleURL = frameworkBundle.url(forResource: "espeak-ng-data", withExtension: "bundle") {
            return dataBundleURL.path
        }
        return nil
    }
    
    static func test_espeak_terminate_without_initialize() {
        log("testing espeak_Terminate without espeak_Initialize\n")
        log("\(espeak_Terminate() == EE_OK)")
    }

    static func test_espeak_initialize() {
        log("testing espeak_Initialize\n")
        
        if let bundleURLStr = findDataBundlePath() {
            log("\(espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, bundleURLStr, 0) == 22050)")
            log("\(espeak_Terminate() == EE_OK)")
        } else {
            log("Couldn't find the bundle, cannot test")
        }
    }
    
    static func test_espeak_synth() {
        log("testing espeak_Synth\n")
        log("\(espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, findDataBundlePath()!, 0) == 22050)")
        
        let test = "One two three.";
        log("\(espeak_Synth(test, strlen(test) + 1, 0, POS_CHARACTER, 0, UInt32(espeakCHARS_AUTO), nil, nil) == EE_OK)")
        log("\(espeak_Synchronize() == EE_OK)")
        log("\(espeak_Terminate() == EE_OK)")
    }
    
    static func test_espeak_ng_phoneme_events(enabled: Int, ipa: Int) {
        log("testing espeak_ng_SetPhonemeEvents(enabled=\(enabled), ipa=\(ipa))")
            
        espeak_ng_InitializePath(findDataBundlePath()!);
        var context: espeak_ng_ERROR_CONTEXT?
        espeak_ng_Initialize(&context)
        espeak_ng_InitializeOutput(espeak_ng_OUTPUT_MODE(0), 0, nil)
        espeak_SetSynthCallback(_test_espeak_ng_phoneme_events_cb)
        espeak_ng_SetPhonemeEvents(Int32(enabled), Int32(ipa))

        let phoneme_events = UnsafeMutablePointer<UInt8>.allocate(capacity: 256)
        let test = "test";

        espeak_ng_Synthesize(test, strlen(test) + 1, 0, POS_CHARACTER, 0, UInt32(espeakCHARS_AUTO), nil, phoneme_events)
        espeak_ng_Synchronize()
        
        if enabled > 0 {
            if ipa > 0 {
                let phonemeEventsData = NSData(bytes: phoneme_events, length: 256)
                let phonemeEventsStr = String(data: phonemeEventsData as Data, encoding: String.Encoding.utf8)?.replacingOccurrences(of: "\0", with: "")
                if phonemeEventsStr == "t ˈɛ s t  " {
                    log("true")
                } else {
                    log("false")
                }
            } else {
                let phonemeEventsData = NSData(bytes: phoneme_events, length: 256)
                let phonemeEventsStr = String(data: phonemeEventsData as Data, encoding: String.Encoding.utf8)?.replacingOccurrences(of: "\0", with: "")
                if phonemeEventsStr == "t 'E s t _: _" {
                    log("true")
                } else {
                    log("false")
                }
            }
        } else {
            log("\(phoneme_events[0] == 0)")
        }
        log("\(espeak_Terminate() == EE_OK)")
    }
}
