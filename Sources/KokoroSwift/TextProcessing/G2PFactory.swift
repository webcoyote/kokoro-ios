public enum G2P {
  case misaki
  case eSpeakNG
}

final class G2PFactory {
  enum G2PError: Error {
    case noSuchEngine
  }
  
  static func createG2PProcessor(engine: G2P) throws -> G2PProcessor {
    switch engine {
    
    case .misaki:
#if canImport(MisakiSwift)
      return MisakiG2PProcessor()
#else
      throw G2PError.noSuchEngine
#endif

    case .eSpeakNG:
#if canImport(eSpeakNGLib)
      return eSpeakNGG2PProcessor()
#else
      throw G2PError.noSuchEngine
#endif
    }
  }
}
