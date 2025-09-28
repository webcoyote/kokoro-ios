enum G2PProcessorError : Error {
  case processorNotInitialized
  case unsupportedLanguage
}

protocol G2PProcessor {
  func setLanguage(_ language: Language) throws
  func process(input: String) throws -> String
}
