"""Extract figures and tables from the study's ten-page paper using Poppler.

Crop coordinates refer to the supplied paper layout, measured at 120 dpi.
The source PDF stays local; the manifest records its hash without its filename.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
TITLE = "Exploring Jailbreak Vulnerability Across Depth-Based Dynamic Inference in Large Language Models"
# (kind, number, page, output stem, x, y, width, height), at 120 dpi.
CROPS = (
    ("figure", 1, 2, "loss_vs_asr", 180, 670, 660, 255),
    ("figure", 2, 5, "pipeline", 202, 496, 616, 273),
    ("figure", 3, 6, "attack_success", 185, 462, 655, 304),
    ("figure", 4, 7, "loss_curves", 180, 115, 690, 460),
    ("table", 1, 5, "model_configuration", 180, 126, 665, 345),
    ("table", 2, 7, "utility_clean", 180, 746, 665, 264),
    ("table", 3, 8, "utility_attacked", 180, 127, 665, 287),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    if not args.paper.is_file():
        parser.error("paper PDF does not exist")
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        parser.error("install Poppler to provide pdftoppm")

    assets = []
    for kind, number, page, name, x, y, width, height in CROPS:
        directory = REPO / ("paper_figures" if kind == "figure" else "paper_tables") / "published"
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / name
        subprocess.run([
            renderer, "-f", str(page), "-singlefile", "-r", "240",
            "-x", str(x * 2), "-y", str(y * 2),
            "-W", str(width * 2), "-H", str(height * 2),
            "-png", str(args.paper), str(prefix),
        ], check=True)
        output = prefix.with_suffix(".png").relative_to(REPO).as_posix()
        assets.append({"kind": kind, "number": number, "page": page,
                       "crop_at_120_dpi": [x, y, width, height], "output": output})
        print(output)
    manifest = {
        "paper_title": TITLE,
        "source_sha256": hashlib.sha256(args.paper.read_bytes()).hexdigest(),
        "render_dpi": 240,
        "assets": assets,
    }
    (REPO / "paper_figures/published/source.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
