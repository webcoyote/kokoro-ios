import AVFoundation
import MLX
import SwiftUI

class MLXTestModel: ObservableObject {
  let kokoroTTSEngine: KokoroTTS!
  let audioEngine: AVAudioEngine!
  let playerNode: AVAudioPlayerNode!

  init() {
    kokoroTTSEngine = KokoroTTS()
    audioEngine = AVAudioEngine()
    playerNode = AVAudioPlayerNode()
    audioEngine.attach(playerNode)
  }

  func say(_ text: String) {
    let mainTimer = BenchmarkTimer.shared.create(id: "TTSGeneration")
    let audioBuffer = try! kokoroTTSEngine.generateAudio(voice: .afHeart, text: text)
    BenchmarkTimer.shared.stop(id: "TTSGeneration")
    BenchmarkTimer.shared.printLog(id: "TTSGeneration")

    BenchmarkTimer.shared.reset()

    let audio = audioBuffer[0].asArray(Float.self)

    let sampleRate = 24000.0
    let audioLength = Double(audio.count) / sampleRate
    print("Audio length: " + String(format: "%.4f", audioLength))

    print("\(mainTimer!.deltaTime)")
    print("Speed: " + String(format: "%.2f", audioLength / mainTimer!.deltaTime))

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)) else {
      print("Couldn't create buffer")
      return
    }

    buffer.frameLength = buffer.frameCapacity
    let channels = buffer.floatChannelData!
    for i in 0 ..< audio.count {
      channels[0][i] = audio[i]
    }

    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
    do {
      try audioEngine.start()
    } catch {
      print("Audio engine failed to start: \(error.localizedDescription)")
      return
    }

    playerNode.scheduleBuffer(buffer, at: nil, options: .interrupts, completionHandler: nil)
    playerNode.play()
  }
}
