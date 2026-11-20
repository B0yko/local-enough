"""Render the mixed-workload router table as a terminal-style demo GIF.

The table is real CLI output: the script runs `local-enough route --simulate --run reference`
and animates it. Rerun after any change to the reference run:

    uv run python scripts/make_demo_gif.py --out docs/img/demo.gif

Only needs Pillow (already a dependency via matplotlib) and the DejaVu Sans Mono font that
matplotlib ships, so it works on macOS and Linux.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

BG = (13, 17, 23)  # GitHub dark dimmed
FG = (230, 237, 243)
MUTED = (139, 148, 158)
GREEN = (63, 185, 80)
FONT_SIZE = 15
LINE_H = 24
PAD = 24
TITLE = "local-enough — mixed workload, reference run (offline replay)"
COMMAND = "local-enough route --simulate --run reference"
HIGHLIGHT = "76.9"  # the gated router's saving vs all-frontier, from the table itself


def run_command() -> list[str]:
    proc = subprocess.run(
        ["local-enough", "route", "--simulate", "--run", "reference"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.rstrip() for line in proc.stdout.strip().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/img/demo.gif"))
    args = parser.parse_args()

    lines = run_command()
    if not any(HIGHLIGHT in line for line in lines):
        sys.exit(f"expected a row containing {HIGHLIGHT!r} in the simulate table; did the reference run change?")

    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    bold = ImageFont.truetype(str(font_path), FONT_SIZE)
    char_w = font.getlength("M")
    width = int(max(len(TITLE), len(COMMAND) + 2, max(len(line) for line in lines)) * char_w) + 2 * PAD
    height = PAD + LINE_H + 8 + LINE_H * (len(lines) + 2) + PAD

    def frame(typed: str, shown: int) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.text((PAD, PAD // 2), TITLE, font=bold, fill=MUTED)
        y = PAD + LINE_H + 8
        d.text((PAD, y), "$", font=bold, fill=GREEN)
        d.text((PAD + 2 * char_w, y), typed, font=font, fill=FG)
        y += LINE_H * 2
        for i, line in enumerate(lines[:shown]):
            color = GREEN if HIGHLIGHT in line else (MUTED if i == 0 else FG)
            d.text((PAD, y), line, font=font, fill=color)
            y += LINE_H
        return img

    frames: list[Image.Image] = []
    durations: list[int] = []
    # Type the command.
    for n in range(0, len(COMMAND) + 1, 3):
        frames.append(frame(COMMAND[:n], 0))
        durations.append(45)
    frames.append(frame(COMMAND, 0))
    durations.append(350)
    # Print the table line by line.
    for shown in range(1, len(lines) + 1):
        frames.append(frame(COMMAND, shown))
        durations.append(160)
    # Hold the result.
    frames.append(frame(COMMAND, len(lines)))
    durations.append(4500)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB, {len(frames)} frames, {width}x{height})")


if __name__ == "__main__":
    main()
