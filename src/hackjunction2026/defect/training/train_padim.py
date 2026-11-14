"""Scenario 1: PaDiM. Simplest and fastest (one pass over the data, ~1 min); the baseline.

    uv run --extra anomalib python -m hackjunction2026.defect.training.train_padim --dataset v1

A Gaussian per image patch over ResNet-18 features. Small model, likely to run on the NPU.
Weaker than PatchCore on small defects.
"""

from anomalib.models import Padim

from hackjunction2026.defect.training.common import run_scenario


def build(image_size: int) -> Padim:
    return Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_processor=Padim.configure_pre_processor(image_size=(image_size, image_size)),
    )


def main(argv: list[str] | None = None) -> None:
    run_scenario("padim", build, batch_size=16, notes="PaDiM resnet18, baseline", argv=argv)


if __name__ == "__main__":
    main()
