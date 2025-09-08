import SwiftUI
import MLX

@main
struct mlxtestApp: App {
  let model = MLXTestModel()
    
  init() {
    MLX.GPU.set(cacheLimit: 50 * 1024 * 1024)
    MLX.GPU.set(memoryLimit: 900 * 1024 * 1024)
  }

  var body: some Scene {
    WindowGroup {
      ContentView(viewModel: model)
    }
  }
}
