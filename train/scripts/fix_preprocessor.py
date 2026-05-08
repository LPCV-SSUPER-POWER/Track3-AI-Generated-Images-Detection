#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: fix_preprocessor.py <model_dir>", file=sys.stderr)
        return 2

    model_dir = Path(sys.argv[1])
    config_path = model_dir / "preprocessor_config.json"
    if not config_path.exists():
        print(f"missing: {config_path}", file=sys.stderr)
        return 1

    data = json.loads(config_path.read_text())
    size = data.get("size")
    if not isinstance(size, dict):
        print(f"unexpected size field in {config_path}", file=sys.stderr)
        return 1

    changed = False
    if "shortest_edge" not in size and "min_pixels" in size:
        size["shortest_edge"] = size["min_pixels"]
        changed = True
    if "longest_edge" not in size and "max_pixels" in size:
        size["longest_edge"] = size["max_pixels"]
        changed = True
    if "min_pixels" in size:
        size.pop("min_pixels")
        changed = True
    if "max_pixels" in size:
        size.pop("max_pixels")
        changed = True

    if changed:
        data["size"] = size
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(f"patched {config_path}")
    else:
        print(f"ok {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
