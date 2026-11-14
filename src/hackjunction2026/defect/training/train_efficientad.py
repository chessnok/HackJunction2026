"""Scenario 3: EfficientAd (small). Very fast at inference, good for the latency score.

    uv run --extra anomalib python -m hackjunction2026.defect.training.train_efficientad --dataset v1

A small student network learns to imitate a pretrained teacher on normal images; where it fails, the
image is anomalous. Trains for many steps with batch size 1 (tens of minutes on the iGPU). The first
run downloads ImageNette (~1.5 GB, into datasets/imagenette, git-ignored) and pretrained weights.
Untested so far (not run on the dev machine because of the download).
"""

from anomalib.models import EfficientAd

from hackjunction2026.defect.training.common import run_scenario


def build(image_size: int) -> EfficientAd:
    return EfficientAd(
        imagenet_dir="datasets/imagenette",
        model_size="small",
        pre_processor=EfficientAd.configure_pre_processor(image_size=(image_size, image_size)),
    )


def main(argv: list[str] | None = None) -> None:
    run_scenario("efficientad", build, batch_size=1, notes="EfficientAd small", argv=argv)


if __name__ == "__main__":
    main()
