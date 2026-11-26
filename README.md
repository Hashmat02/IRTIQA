# IRTIQA

[![arXiv](https://img.shields.io/badge/arXiv-2611.34303-b31b1b.svg)](https://arxiv.org/abs/2611.34303)
[![Findings of EACL 2027](https://img.shields.io/badge/Findings%20of-EACL%202027-blue.svg)](https://arxiv.org/abs/2611.34303)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**IRTIQA: Evolving in English, Evaluated in Urdu — Does Agent Self-Evolution
Transfer from English to Urdu and Code-Switched Input?**

Abdullah Hashmat, Danish Javed, Usman Naseem, Agha Ali Raza
Findings of the Association for Computational Linguistics: EACL 2027
📄 [Paper](https://arxiv.org/abs/2611.34303)

IRTIQA ("evolution" / "gradual ascent" in Urdu) is a benchmark for studying
whether self-evolving LLM agents, trained and evaluated almost exclusively in
English, retain their capability gains when the input language changes. It
contains aligned **English**, **Urdu-translated**, and **Urdu-English
code-switched** variants of DROP, SNIPS, AppWorld, and SkillsBench, built with
a matrix-language code-switching pipeline and human verification. Every
instance across the three conditions shares the same gold answer or
executable goal, so any performance change is attributable to the language
transformation alone.

## Abstract

As LLM agents increasingly learn from their own experience, the tasks they
learn from and are evaluated on are almost always in English. We ask whether
self-evolution transfers from English to Urdu, and release IRTIQA, a
benchmark of aligned English, Urdu-translated, and Urdu code-switched variants
of DROP, SNIPS, AppWorld, and SkillsBench, built with a matrix-language
code-switching pipeline and human verification. Using IRTIQA, we evaluate
three self-evolving frameworks, Tool-R0 (self-play RL), AgentEvolver
(experience-guided RL), and Autogenesis (protocol-level evolution), against
their unevolved backbones. Self-evolution often raises capability:
AgentEvolver lifts AppWorld goal completion from near zero to 20–36% across
all conditions, and Autogenesis improves SkillsBench pass rate from 16.48% to
22.22%. But it rarely closes the language gap. Evolved systems remain
stronger in English, by up to 0.26 DROP F1 and 30 AppWorld goal points, the
gap is largest on multi-step tasks, and code-switching behaves as a distinct
condition rather than degraded translation. On a shared Gemma-4 31B backbone,
the gap exists before evolution, and RL-based evolution can hurt Urdu
reasoning more than English. Language parity must therefore be an explicit
design target, as it does not emerge from self-improvement alone.

## Key Findings

- **Capability gains don't erase the language gap.** Self-evolution mainly
  strengthens capabilities the backbone already has; evolved systems stay
  stronger in English on DROP, AppWorld, and SkillsBench across all three
  frameworks.
- **The gap predates evolution.** On a shared Gemma-4 31B backbone, English
  outperforms Urdu before any framework is applied, and no framework closes
  it — two RL frameworks (Tool-R0, AgentEvolver) even lower DROP F1 more in
  Urdu than in English relative to the base model.
- **Code-switching ≠ degraded translation.** Code-switched input often
  outperforms pure translation (e.g., it anchors entities and tool arguments
  via retained English terms), but the direction flips for some
  framework–task pairs, so the two conditions must be reported separately.
- **Gaps widen with task horizon.** Near-ceiling on SNIPS classification but
  largest on multi-step AppWorld and SkillsBench tasks, consistent with
  compounding errors in longer-horizon execution.

## Contents

- `datasets/`: final aligned benchmark data (English / Urdu / code-switched)
- `data_curation/`: sampling, translation, code-switching, and cleanup scripts
- `annotation_ui/`: human verification UI
- `frameworks/`: references and portable wrappers for the evaluated
  self-evolving frameworks
- `scripts/`: paper evaluation runners

## Datasets

| Dataset | Size per variant |
|---|---:|
| DROP | 10,000 examples |
| SNIPS | 5,000 instances |
| AppWorld | 168 tasks |
| SkillsBench | 87 tasks |

Variants:

- `english`
- `urdu`
- `codeswitched`

## Paper Evaluation Matrix

| Framework | DROP | SNIPS | AppWorld | SkillsBench |
|---|---:|---:|---:|---:|
| Tool-R0 | yes | yes | yes | no |
| AgentEvolver | yes | yes | yes | no |
| Autogenesis | yes | yes | yes | yes |

All three frameworks are also re-instantiated on a shared `google/gemma-4-31B-it`
backbone to separate framework effects from native-backbone differences (see
Section 5.2 of the paper).

## Running

Generic runner:

```bash
bash scripts/run_framework.sh \
  --framework toolr0 \
  --dataset snips \
  --variant urdu
```

Convenience runners exist only for paper-reported combinations:

```bash
bash scripts/run_toolr0_drop.sh --variant english
bash scripts/run_toolr0_snips_multilang.sh --variant urdu
bash scripts/run_toolr0_appworld_multilang.sh --variant codeswitched

bash scripts/run_agentevolver_drop.sh --variant english
bash scripts/run_agentevolver_snips.sh --variant urdu
bash scripts/run_agentevolver_appworld.sh --variant codeswitched

bash scripts/run_autogenesis_drop.sh --variant english
bash scripts/run_autogenesis_snips.sh --variant urdu
bash scripts/run_autogenesis_appworld.sh --variant codeswitched
bash scripts/run_autogenesis_skillsbench.sh --variant english
```

Set framework-specific paths locally, for example:

- `TOOLR0_INTEGRATION=/path/to/Tool-R0/arr_integration`
- `AGENTEVOLVER_ROOT=/path/to/AgentEvolver`
- `AUTOGENESIS_ROOT=/path/to/Autogenesis`
- `MODEL_BASE_URL=http://localhost:8000/v1`
- `MODEL_NAME=<served-model-name>`

Machine-specific paths, logs, results, caches, checkpoints, and personal
annotation outputs are intentionally excluded.

## Citation

If you use IRTIQA, please cite:

```bibtex
@misc{hashmat2026irtiqaevolvingenglishevaluated,
      title={IRTIQA: Evolving in English, Evaluated in Urdu -- Does Agent Self-Evolution Transfer from English to Urdu and Code-Switched Input?}, 
      author={Abdullah Hashmat and Danish Javed and Usman Naseem and Agha Ali Raza},
      year={2026},
      eprint={2611.34303},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.34303}, 
}
```

Accepted to Findings of the Association for Computational Linguistics: EACL
2027. The arXiv entry above is the current citable version; this will be
swapped for the official ACL Anthology BibTeX once it's assigned.

## License

Code in this repository is released under the [MIT License](LICENSE).
Released dataset variants are derived from DROP, SNIPS, AppWorld, and
SkillsBench and remain subject to each source benchmark's own license; see
`datasets/README.md` for provenance.

## Contact

Questions or issues: open a GitHub issue, or email
abdullah.hashmat02@gmail.com.
