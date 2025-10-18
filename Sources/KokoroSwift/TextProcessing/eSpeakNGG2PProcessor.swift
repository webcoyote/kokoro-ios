#if canImport(eSpeakNGLib)

import Foundation
import eSpeakNGLib

final class eSpeakNGG2PProcessor : G2PProcessor {
  private var eSpeakEngine: eSpeakNG?

  func setLanguage(_ language: Language) throws {
    eSpeakEngine = try eSpeakNG()
    
    if let language = eSpeakNG.Language(rawValue: language.rawValue), let eSpeakEngine {
      try eSpeakEngine.setLanguage(language: language)
    } else {
      throw G2PProcessorError.unsupportedLanguage
    }
  }
  
  func process(input: String) throws -> String {
    guard let eSpeakEngine else { throw G2PProcessorError.processorNotInitialized }
    return try eSpeakEngine.phonemize(text: input)
  }
}

#endif
