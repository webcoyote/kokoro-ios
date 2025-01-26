import os

let logger = Logger(subsystem: "com.kokoro.espeak-ng-test", category: "app")

func log(_ message: String) {
    logger.log("\(message)")
}

