import torch.onnx
import onnx
from tonality_model import *

'''
------------ Export Model to ONNX -------------
'''
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move to GPU
model = KeyEstimationCNN().to(device)
checkpoint = torch.load('best_tonality_model_rtk.pth')
model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)
model.eval()

# 1. Define a dummy input that matches the input shape your model expects.
# This should have the same dimensions as the input during inference.
dummy_input = torch.randn(1, 1, 103, 175).to(device)  # Adjust size based on your actual input

# 2. Export the model to ONNX format.
onnx_file_path = "tonality_model.onnx"  # Specify the output file path

# Export the model
torch.onnx.export(
    model,                            # The model to export
    dummy_input,                      # A dummy input to specify the input shape
    onnx_file_path,                   # Path where the ONNX file will be saved
    input_names=['input'],            # Name the input layer (optional)
    output_names=['output'],          # Name the output layer (optional)
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Dynamic batch size
    opset_version=10,                 # ONNX opset version
    verbose=True                      # Display detailed export info
)

print(f"Model successfully exported to {onnx_file_path}")

# '''
# ----------- Verify the export -------------
# '''
# # Load the ONNX model
# onnx_file_path = "tonality_model.onnx"  # Specify the output file path
# onnx_model = onnx.load(onnx_file_path)

# # Check that the model is well-formed
# onnx.checker.check_model(onnx_model)

# print("ONNX model is valid!")