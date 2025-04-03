# Kokoro TTS for Swift

Experimental implementation of Kokoro TTS for macOS and iOS devices using MLX Swift and eSpeak NG.

Kokoro TTS port is based on the great work done in [MLX-Audio project](https://github.com/Blaizzy/mlx-audio), where the model was ported from PyTorch to MLX Python. This project ports the MLX Python code to MLX Swift.

The project uses eSpeak NG as a phonemizer, which is different from what the original Kokoro TTS uses. This can and will cause differences in the output audio.

Currently the project generates audio ~3.3 times faster than realtime on the release build on iPhone 13 Pro after warm up / first run.

## Running the sample application

Follow these steps to get the sample application running on your iOS device:

1. Get the weights for the model. You can get the weight file from MLX-Audio project or then download directly from Hugging Face: https://huggingface.co/prince-canuma/Kokoro-82M/blob/main/kokoro-v1_0.safetensors
2. Copy the weights model file to the `mlxtest/mlxtest/Resources` directory.
3. Open eSpeak NG project to Xcode.
4. Choose target `espeak-ng .xcframework` and compile. The script creates to project root on `Frameworks` directory `ESpeakNG.xcframework` file that the TTS engine uses for phonemization.
5. Open`mlxtest`, change the bundle identifier and fix the signing for running the app on a device.
6. Run the application!
7. Enter the text to input field and click "Say something". Wait for a while and the audio should be played out loud.

Note that if you want to run the app on a Mac, you can't use iOS emulator because it doesn't support Metal. Just use the default macOS target. For more information, see [MLX Swift documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/running-on-ios).
