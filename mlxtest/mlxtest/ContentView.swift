import SwiftUI

struct ContentView: View {
  @ObservedObject var viewModel: MLXTestModel

  var body: some View {
    VStack {
      Spacer()

      Button {
        viewModel.say("Now I am really starting to like this model because it is fast")
      } label: {
        HStack(alignment: .center) {
          Spacer()
          Text("Say something")
            .foregroundColor(.white)
            .frame(height: 50)
          Spacer()
        }
        .background(.black)
        .padding()
      }

      Spacer()
    }
    .padding()
    .background(.white)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}

#Preview {
  ContentView(viewModel: MLXTestModel())
}
