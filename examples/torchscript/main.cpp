#include <iostream>
#include <memory>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

using namespace std;
int main() {
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("../q_eval_script.pt");

  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  torch::Device device(torch::kCUDA);
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({3, 16, 1024}, device));

  // // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  cout << output.mean(1).argmax(1) << endl;
}
