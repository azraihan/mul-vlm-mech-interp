import json
import shutil
from pathlib import Path

# ── Serial numbers (1-indexed) ────────────────────────────────────────────────

PT_STEERED_WINS  = [16, 23, 61, 68, 76, 116, 190, 200]   # steered ✓, baseline ✗
PT_BASELINE_WINS = [26, 49, 57, 99, 121, 159]             # baseline ✓, steered ✗

RU_STEERED_WINS  = [50, 64, 95, 110, 114, 122, 132, 142, 154, 172, 174, 186, 192, 198]
RU_BASELINE_WINS = [3]

GROUPS = {
    "pt": {
        "json_file":     "data/mmmb/mmmb_pt.json",
        "steered_wins":  PT_STEERED_WINS,
        "baseline_wins": PT_BASELINE_WINS,
    },
    "ru": {
        "json_file":     "data/mmmb/mmmb_ru.json",
        "steered_wins":  RU_STEERED_WINS,
        "baseline_wins": RU_BASELINE_WINS,
    },
}


def export_group(records, dest_dir: Path):
    """Copy images and write questions.json into dest_dir."""
    images_dir = dest_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    output = []
    for record in records:
        src_image = Path(record["image"])          # e.g. data/mmmb/images/pt_0015.jpg
        dest_image = images_dir / src_image.name   # images/pt_0015.jpg

        if src_image.exists():
            shutil.copy2(src_image, dest_image)
        else:
            print(f"  [WARN] image not found: {src_image}")

        output.append({
            "question": record["question"],
            "options":  record["options"],
            "answer":   record["answer"],
            "image":    f"images/{src_image.name}",
        })

    with open(dest_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {len(output)} questions → {dest_dir}")


def main():
    base = Path("mmmb_analysis")

    for lang, cfg in GROUPS.items():
        print(f"\n[{lang.upper()}] Loading {cfg['json_file']} ...")
        with open(cfg["json_file"], encoding="utf-8") as f:
            data = json.load(f)           # list, 0-indexed internally

        for group_name, serials in [
            ("steered_wins",  cfg["steered_wins"]),
            ("baseline_wins", cfg["baseline_wins"]),
        ]:
            records = [data[s - 1] for s in serials]  # convert to 0-indexed
            dest    = base / lang / group_name
            print(f"  → {group_name} ({len(serials)} questions)")
            export_group(records, dest)

    print("\nDone! Output written to: mmmb_analysis/")


if __name__ == "__main__":
    main()