import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: MLXTestModel
  
    var body: some View {
        VStack {
          Spacer()
          
          Button {
            viewModel.say("First time ever I am speaking from mobile device")
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
