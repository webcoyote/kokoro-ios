import UIKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        log("Initializing view")
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // Few simple test cases imported from espeak-ng to verify that espeak-ng works
        log("Starting to test ESpeakNG")
        ESpeakNGTest.test_espeak_terminate_without_initialize()
        ESpeakNGTest.test_espeak_initialize()
        ESpeakNGTest.test_espeak_synth()
        ESpeakNGTest.test_espeak_ng_phoneme_events(enabled: 0, ipa: 0)
        ESpeakNGTest.test_espeak_ng_phoneme_events(enabled: 1, ipa: 0)
        ESpeakNGTest.test_espeak_ng_phoneme_events(enabled: 1, ipa: 1)
    }
}

