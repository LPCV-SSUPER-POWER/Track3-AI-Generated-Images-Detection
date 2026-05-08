#!/usr/bin/env python3
"""패키징 완료 후 zip 검증 (sed로 'py_files' → '$EXP' 치환 후 실행됨)"""
import os
import zipfile
import json
from pathlib import Path

# Self-contained ROOT detection. wrapper가 BUNDLE_ROOT를 export 하면
# /tmp 에 sed-치환 복사본으로 실행되어도 그대로 작동.
QUANT_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = Path(os.environ.get("BUNDLE_ROOT", str(QUANT_ROOT.parent)))

zip_path = str(BUNDLE_ROOT / "submit" / "py_files.zip")
results = str(BUNDLE_ROOT / "results" / "py_files")

all_ok = True

print("=== 1. zip 파일 목록 ===")
with zipfile.ZipFile(zip_path, "r") as zf:
    for info in zf.infolist():
        mb = info.file_size / 1024 / 1024
        print(f"  {info.filename}: {mb:.1f}MB")

    names = [i.filename for i in zf.infolist()]
    required = ["mask.raw", "position_ids_cos.raw", "position_ids_sin.raw",
                "embedding_weights", "tokenizer.json", "inputs.json",
                "veg.serialized.bin", "weight_sharing_model"]
    print()
    print("=== 2. 필수 파일 존재 ===")
    for req in required:
        found = any(req in n for n in names)
        label = "OK" if found else "MISSING"
        if not found:
            all_ok = False
        print(f"  {req}: {label}")

    print()
    print("=== 3. 원본 대비 크기 일치 ===")
    checks = [
        ("SSUPER POWER/mask.raw", f"{results}/Example1A/veg_exports/mask.raw"),
        ("SSUPER POWER/embedding_weights_151936x1536.raw", f"{results}/Example1B/embedding_weights_151936x1536.raw"),
        ("SSUPER POWER/tokenizer.json", f"{results}/Example1B/tokenizer/tokenizer.json"),
        ("SSUPER POWER/serialized_binaries/veg.serialized.bin", f"{results}/Example2A/serialized_binaries/veg.serialized.bin"),
        ("SSUPER POWER/ar128-ar1-cl2048/weight_sharing_model_1_of_1.serialized.bin", f"{results}/Example2B/weight_sharing_model_1_of_1.serialized.bin"),
    ]
    for zip_name, orig_path in checks:
        zip_info = zf.getinfo(zip_name)
        orig_size = os.path.getsize(orig_path)
        match = zip_info.file_size == orig_size
        short = zip_name.rsplit("/", 1)[-1]
        label = "OK" if match else "MISMATCH"
        if not match:
            all_ok = False
        print(f"  {short}: zip={zip_info.file_size} orig={orig_size} [{label}]")

    print()
    print("=== 4. inputs.json 내용 ===")
    with zf.open("SSUPER POWER/inputs.json") as f:
        inp = json.load(f)
    print(f"  embedding_dim: {inp.get('run_veg_embedding_dim')}")
    print(f"  n_tokens: {inp.get('run_veg_n_tokens')}")
    h = inp.get("data_preprocess_inp_h")
    w = inp.get("data_preprocess_inp_w")
    print(f"  inp_h x inp_w: {h} x {w}")
    gc = inp.get("genie_config", {}).get("dialog", {})
    print(f"  max-num-tokens: {gc.get('max-num-tokens')}")
    ctx = gc.get("context", {})
    print(f"  n-vocab: {ctx.get('n-vocab')}")
    print(f"  eos-token: {ctx.get('eos-token')}")

print()
print("=== 5. zip CRC 검증 ===")
with zipfile.ZipFile(zip_path, "r") as zf:
    result = zf.testzip()
    if result is None:
        print("  CRC check: ALL OK")
    else:
        all_ok = False
        print(f"  CRC check: FAILED on {result}")

print()
if all_ok:
    print("=== ALL CHECKS PASSED ===")
else:
    print("=== SOME CHECKS FAILED ===")
