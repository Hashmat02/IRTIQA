# IRTIQA Manifest

Included:

- Final organized dataset index: `datasets`
- Dataset generation scripts: `data_curation`
- Integrated deterministic cleanup logic: `data_curation/code_switch.py`
- Annotation UI source: `annotation_ui/annotate.py`
- Framework references and portable wrappers for Tool-R0, AgentEvolver, and
  Autogenesis
- Paper-matrix runnable script entrypoints: `scripts`

Excluded:

- `logs/`
- `results/`
- `Annotations/`
- `.cache/`
- `.toolr0_tmp/`
- broad framework config trees unrelated to IRTIQA
- unsupported framework/dataset runner scripts
- machine-specific run scripts/configs with absolute paths, GPU partitions,
  cache locations, or model snapshot paths
- framework output folders
- model checkpoints and downloaded external data

The bundle is intentionally symlink-based. Package with care if moving it
outside this repo: preserve symlinks or materialize only the listed targets.
