//
//  kokoro-tts-lib
//
import os

#if DEBUG

let logger = Logger(subsystem: "com.kokoro.kokoro-tts-lib", category: "lib")

func log(_ message: String) {
    logger.log("\(message)")
}

#else

@inline(__always) func log(_ message: String) {}

#endif
