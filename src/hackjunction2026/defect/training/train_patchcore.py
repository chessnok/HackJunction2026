"""Scenario 2: PatchCore. Usually the best quality for this kind of task (one pass, a few minutes).

    uv run --extra anomalib python -m hackjunction2026.defect.training.train_patchcore --dataset v1
    ... --coreset 0.05 --tag coreset005        # try a bigger memory bank

Keeps a "memory bank" of normal patch features and scores how far a new patch is from the nearest
one. The bank grows with the data: --coreset is the share of patches kept. 0.01 keeps the exported
model small and fast (good for the NPU); 0.1 (Anomalib's default) can be hundreds of MB.
"""

import sys

from anomalib.models import Patchcore

from hackjunction2026.defect.training.common import run_scenario


def take_coreset(argv: list[str]) -> tuple[float, list[str]]:
    """Read our own --coreset option and pass the rest on to common.run_scenario."""
    if "--coreset" in argv:
        i = argv.index("--coreset")
        return float(argv[i + 1]), argv[:i] + argv[i + 2 :]
    return 0.01, argv


def main(argv: list[str] | None = None) -> None:
    coreset, rest = take_coreset(list(sys.argv[1:] if argv is None else argv))

    def build(image_size: int) -> Patchcore:
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=coreset,
            pre_processor=Patchcore.configure_pre_processor(image_size=(image_size, image_size)),
        )

    run_scenario("patchcore", build, batch_size=16, notes=f"PatchCore wide_resnet50_2, coreset {coreset}",
                 argv=rest)


if __name__ == "__main__":
    main()
