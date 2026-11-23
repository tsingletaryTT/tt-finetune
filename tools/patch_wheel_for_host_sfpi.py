#!/usr/bin/env python3
"""Patch the tt-metal kernel headers bundled in the tt-finetune wheel so they
compile with a newer host SFPI compiler.

Why this exists
---------------
The v0.2.1 wheel bundles a tt-metal built for SFPI 7.67.0, but tt-metal's JIT
build always compiles device kernels with the host compiler in
/opt/tenstorrent/sfpi. With SFPI 7.96.0 (tt-kmd 2.12.0 hosts) the JIT build of
the first dispatch kernel fails, because the compiler runs with -Werror and
rejects four idioms in the older headers:

  1. `__builtin_rvtt_sfpiadd_i(src, imm, mod1)` is now a deprecated macro that
     warns "provide iptr operands explicitly" (ckernel_sfpu_exp.h,
     ckernel_sfpu_exp2.h).
  2. `sfpi::lut(vFloat, vUInt, vUInt, vUInt)` is deleted; the coefficients must
     be `sfpi::vLut8si` (ckernel_sfpu_sigmoid_appx.h, ckernel_sfpu_tanh.h).

Both fixes keep the generated instructions the same: the macro expands to the
call with the explicit instruction buffer and two zero iptr operands, and a
vLut8si is loaded from the same LReg as the old vUInt.

Usage:  python tools/patch_wheel_for_host_sfpi.py [--check] [--site-packages DIR]
Idempotent. Originals are kept next to each file as `<name>.orig`.
"""
import argparse
import shutil
import sys
from pathlib import Path

LLK_SFPU = "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"

# (file, [(old, new), ...]) -- each `old` must occur exactly `count` times.
PATCHES = [
    ("ckernel_sfpu_exp.h", [(
        "__builtin_rvtt_sfpiadd_i(e.get(), -255, sfpi::SFPIADD_MOD1_CC_LT0)",
        "(__builtin_rvtt_sfpiadd_i)(::ckernel::instrn_buffer, e.get(), -255, 0, 0, sfpi::SFPIADD_MOD1_CC_LT0)",
        1)]),
    ("ckernel_sfpu_exp2.h", [(
        "__builtin_rvtt_sfpiadd_i(e.get(), -255, sfpi::SFPIADD_MOD1_CC_LT0)",
        "(__builtin_rvtt_sfpiadd_i)(::ckernel::instrn_buffer, e.get(), -255, 0, 0, sfpi::SFPIADD_MOD1_CC_LT0)",
        1)]),
    ("ckernel_sfpu_sigmoid_appx.h", [
        ("    vUInt l0 = l_reg[LRegs::LReg0];\n    vUInt l1 = l_reg[LRegs::LReg1];\n    vUInt l2 = l_reg[LRegs::LReg2];\n",
         "    sfpi::vLut8si l0 = l_reg[LRegs::LReg0];\n    sfpi::vLut8si l1 = l_reg[LRegs::LReg1];\n    sfpi::vLut8si l2 = l_reg[LRegs::LReg2];\n", 1)]),
    ("ckernel_sfpu_tanh.h", [
        ("        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];\n        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];\n        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];\n",
         "        sfpi::vLut8si l0 = l_reg[sfpi::LRegs::LReg0];\n        sfpi::vLut8si l1 = l_reg[sfpi::LRegs::LReg1];\n        sfpi::vLut8si l2 = l_reg[sfpi::LRegs::LReg2];\n", 1)]),
]


def find_runtime(site_packages: Path | None) -> Path:
    if site_packages is None:
        import importlib.util
        spec = importlib.util.find_spec("tt_finetune")  # does not open the device
        if spec is None or spec.origin is None:
            sys.exit("tt_finetune is not installed in this interpreter; pass --site-packages")
        site_packages = Path(spec.origin).parent / "_runtime" / "site-packages"
    root = site_packages / "pjrt_plugin_tt" / "tt-metal"
    if not (root / LLK_SFPU).is_dir():
        sys.exit(f"no bundled tt-metal kernel headers under {root}")
    return root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site-packages", type=Path)
    ap.add_argument("--check", action="store_true", help="report only; change nothing")
    args = ap.parse_args()
    root = find_runtime(args.site_packages)
    status = 0
    for name, edits in PATCHES:
        path = root / LLK_SFPU / name
        text = path.read_text()
        new = text
        for old, repl, count in edits:
            if old in new:
                if new.count(old) != count:
                    sys.exit(f"{name}: expected {count} match(es), found {new.count(old)}")
                new = new.replace(old, repl)
        # Patched file: no old idiom left. Also catch the lut call still taking vUInt.
        if new == text:
            print(f"{name}: already patched or not applicable")
            continue
        print(f"{name}: {'needs patch' if args.check else 'patched'}")
        status = 1 if args.check else status
        if not args.check:
            if not path.with_name(name + ".orig").exists():
                shutil.copy2(path, path.with_name(name + ".orig"))
            path.write_text(new)
    return status


if __name__ == "__main__":
    sys.exit(main())
