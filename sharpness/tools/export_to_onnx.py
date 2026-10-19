import torch

from common.paths import ensure_dir
from sharpness.config import IMAGE_SIZE, BEST_CHECKPOINT, ONNX_PATH
from sharpness.model import get_efficientnet


def export_efficientnet_dynamo():
    DEVICE = torch.device("cpu")

    print("Loading model...")
    weights_path = BEST_CHECKPOINT

    model = get_efficientnet(DEVICE, weights_path=str(weights_path), eval_mode=True)
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    ensure_dir(ONNX_PATH.parent)
    onnx_file_path = str(ONNX_PATH)
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
