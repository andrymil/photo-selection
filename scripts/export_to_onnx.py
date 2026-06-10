import torch
from utils.model import get_efficientnet

EXPERIMENT = "EfficientNetV2S_384px_batch16_with_EBB_BEST"


def export_efficientnet_dynamo():
    DEVICE = torch.device("cpu")

    print("Loading model...")
    weights_path = f"models/checkpoints/{EXPERIMENT}.pth"

    model = get_efficientnet(DEVICE, weights_path=str(weights_path), eval_mode=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 384, 384, device=DEVICE)

    onnx_file_path = "models/onnx/blur_classifier.onnx"
    print(f"Exporting to ONNX...")

    onnx_program = torch.onnx.export(
        model,
        args=(dummy_input,),
        f=onnx_file_path,
        dynamo=True,
    )

    print(f"Success! Model saved to: {onnx_file_path}")

    print("\nInputs and outputs names of the graph:")
    print(onnx_program.model_proto.graph.input[0].name)
    print(onnx_program.model_proto.graph.output[0].name)


if __name__ == "__main__":
    export_efficientnet_dynamo()
