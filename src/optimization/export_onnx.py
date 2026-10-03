from pathlib import Path

import torch

from src.models.resnet50_finetune import create_resnet50_finetune


MODEL_PATH = Path("models/resnet50_finetune.pth")
OUTPUT_PATH = Path("models/resnet50_finetune.onnx")


def main():
    device = torch.device("cpu")

    model = create_resnet50_finetune(
        num_classes=6
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    example_input = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    onnx_program = torch.onnx.export(
        model,
        (example_input,),
        input_names=["input"],
        output_names=["logits"],
        dynamo=True,
    )

    onnx_program.save(
        OUTPUT_PATH
    )

    print(
        "ONNX model saved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()