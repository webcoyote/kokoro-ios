# Kokoro TTS for iOS

Experimental implementation of Kokoro TTS for iOS devices using MLX Swift and eSpeak NG.

Kokoro TTS port is based on the great work done in [MLX-Audio project](https://github.com/Blaizzy/mlx-audio), where the Kokoro TTS model was ported from PyTorch to MLX Python. This project ports the MLX Python code to MLX Swift so that the model can be run in iOS devices.

As a phonemizer this project uses eSpeak NG, which differs from what original Kokoro TTS uses. This might and will create differences to the output audio.

Currently the project generates audio around ~3.3 times faster than real-time on iPhone 13 Pro on release build after warm up / first run.

## Running the example app

Follow these steps to get the example app running on your iOS device:

1. Get the weights for the model. You can get the weight file from MLX-Audio project or then download directly from Hugging Face: https://huggingface.co/prince-canuma/Kokoro-82M/blob/main/kokoro-v1_0.safetensors
2. Copy the weights model file to the `mlxtest/mlxtest/Resources` directory
3. Open eSpeak NG project to Xcode
4. Choose target "espeak-ng .xcframework" and compile. The script creates to project root on `Frameworks` directory `ESpeakNG.xcframework` file that the TTS engine uses for phonemization
5. Open`mlxtest`, change the bundle identifier and fix the signing for running the app on a device
6. Run the application!
6. After clicking "Say something" wait for a while and the audio should be played out loud

Note that if you want to run the app on Mac, you can't use iOS emulator because it doesn't support Metal. You can run the app on "Mac (Designed for iPad)" destination. Further information available on [MLX Swift documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/running-on-ios)
