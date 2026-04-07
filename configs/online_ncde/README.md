# online_ncde config layout

`configs/online_ncde/` now uses fast/slow-system combinations as the main unit:

- `fast_<fast_system>__slow_<slow_system>/`: canonical config directory for one fast/slow pair.
- Each combo directory keeps self-contained YAML files. Different combos should not share a `base.yaml`.
- Root-level `train.yaml`, `eval.yaml`, `base.yaml`, and `base_1s_tminus1.yaml` are compatibility entrypoints that point to the current default combo.

Current default combo:

- `fast_opusv1t__slow_opusv2l/`

Recommended rule for future additions:

1. Copy the closest existing combo directory.
2. Rename it to `fast_<fast_system>__slow_<slow_system>/`.
3. Update only the paths/variants/output dirs inside that directory.
4. Keep shared dataset metadata files (for example `ncde_align_infos_*.pkl`) at the root of `configs/online_ncde/` unless a combo truly needs its own copy.

This keeps each fast/slow pair isolated while preserving old script defaults.
