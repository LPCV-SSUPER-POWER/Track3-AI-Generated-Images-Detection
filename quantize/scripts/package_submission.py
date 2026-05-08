#!/usr/bin/env python3
"""
제출 zip 패키징 스크립트
사용법: python package_submission.py --exp-name FINAL_RESULTS --team-name "SSUPER POWER"
출력: {BUNDLE_ROOT}/submit/{exp_name}.zip

ROOT는 환경변수 BUNDLE_ROOT 우선, 없으면 스크립트 자기 위치 기준 자동 감지.
"""
import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

# Self-contained ROOT detection
QUANT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> quantize/
BUNDLE_ROOT = Path(os.environ.get("BUNDLE_ROOT", str(QUANT_ROOT.parent)))


def copy_required(src: Path, dst: Path, label: str):
    if not src.exists():
        raise FileNotFoundError(f"Missing {label}: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {label}: {src.name} ({src.stat().st_size // 1024 // 1024}MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True, help="실험명 (e.g. FINAL_RESULTS)")
    parser.add_argument("--team-name", default="SSUPER POWER", help="팀 이름")
    parser.add_argument("--llm-folder", default="ar128-ar1-cl2048", help="LLM binary 폴더명")
    args = parser.parse_args()

    results = BUNDLE_ROOT / "results" / args.exp_name
    submit_dir = BUNDLE_ROOT / "submit"
    team_dir = submit_dir / args.team_name

    # 기존 제출 디렉토리 정리
    if team_dir.exists():
        shutil.rmtree(team_dir)
    team_dir.mkdir(parents=True, exist_ok=True)

    # 산출물 경로
    veg_exports = results / "Example1A" / "veg_exports"
    llm_dir = results / "Example1B"
    veg_bin = results / "Example2A" / "serialized_binaries" / "veg.serialized.bin"
    ws_bin = results / "Example2B" / "weight_sharing_model_1_of_1.serialized.bin"
    inputs_json = QUANT_ROOT / "inference" / "contestant_uploads" / "inputs.json"

    # 복사
    print(f"[package] exp={args.exp_name}, team={args.team_name}")
    print(f"[package] BUNDLE_ROOT: {BUNDLE_ROOT}")
    print(f"[package] results: {results}")
    print(f"[package] output:  {submit_dir}")
    print()

    copy_required(veg_exports / "mask.raw", team_dir / "mask.raw", "mask")
    copy_required(veg_exports / "position_ids_cos.raw", team_dir / "position_ids_cos.raw", "pos_cos")
    copy_required(veg_exports / "position_ids_sin.raw", team_dir / "position_ids_sin.raw", "pos_sin")

    # embedding (파일명에 dimension 포함)
    emb_files = sorted(llm_dir.glob("embedding_weights*.raw"))
    if not emb_files:
        raise FileNotFoundError(f"Missing embedding_weights*.raw in {llm_dir}")
    copy_required(emb_files[0], team_dir / emb_files[0].name, "embedding")

    copy_required(llm_dir / "tokenizer" / "tokenizer.json", team_dir / "tokenizer.json", "tokenizer")
    copy_required(veg_bin, team_dir / "serialized_binaries" / "veg.serialized.bin", "veg_bin")
    copy_required(ws_bin, team_dir / args.llm_folder / "weight_sharing_model_1_of_1.serialized.bin", "llm_bin")
    copy_required(inputs_json, team_dir / "inputs.json", "inputs")

    # zip 생성
    zip_path = submit_dir / f"{args.exp_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(team_dir.rglob("*")):
            if path.is_file():
                arcname = str(Path(args.team_name) / path.relative_to(team_dir))
                zf.write(path, arcname=arcname)

    print()
    print(f"[package] zip: {zip_path}")
    print(f"[package] size: {zip_path.stat().st_size // 1024 // 1024}MB")

    # zip 내용 확인
    print()
    print("[package] zip 내용:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size // 1024 // 1024}MB)")


if __name__ == "__main__":
    main()
