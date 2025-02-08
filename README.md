# Kokoro TTS for iOS

Experimental implementation of Kokoro TTS for iOS devices using CoreML.

Currently futher work is blocked because of these two CoreML Tools issues that prevent to convert parts of the model to CoreML:
- [torch.nn.ConvTranspose1d Conversion failed](https://github.com/apple/coremltools/issues/1946)
- [Adding support for complex numbers (e.g. 'torch.complex', 'torch.view_as_real', etc.) for PyTorch](https://github.com/apple/coremltools/issues/1539)
