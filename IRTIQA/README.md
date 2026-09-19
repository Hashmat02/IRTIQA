# IRTIQA

IRTIQA is a benchmark for multilingual self-evolving agents. It contains aligned
English, Urdu-translated, and Urdu-English code-switched variants of DROP,
SNIPS, AppWorld, and SkillsBench. Each aligned instance preserves the same gold
answer across all language conditions.

## Contents

- `datasets/`: final aligned benchmark data
- `data_curation/`: sampling, translation, code-switching, and cleanup scripts
- `annotation_ui/`: human verification UI
- `scripts/`: paper evaluation runners

## Datasets

| Dataset | Size per condition |
|---|---:|
| DROP | 10,000 examples |
| SNIPS | 6,400 examples |
| AppWorld | 732 tasks |
| SkillsBench | 87 tasks |

Conditions:

- `english`
- `urdu`
- `codeswitched`

## Paper Evaluation Matrix

| Framework | DROP | SNIPS | AppWorld | SkillsBench |
|---|---:|---:|---:|---:|
| Tool-R0 | yes | yes | yes | no |
| AgentEvolver | yes | yes | yes | no |
| Autogenesis | yes | yes | yes | yes |

## Running

Generic runner:

```bash
bash IRTIQA/scripts/run_framework.sh \
  --framework toolr0 \
  --dataset snips \
  --variant urdu
```

Convenience runners exist only for paper-reported combinations:

```bash
bash IRTIQA/scripts/run_toolr0_drop.sh --variant english
bash IRTIQA/scripts/run_toolr0_snips_multilang.sh --variant urdu
bash IRTIQA/scripts/run_toolr0_appworld_multilang.sh --variant codeswitched

bash IRTIQA/scripts/run_agentevolver_drop.sh --variant english
bash IRTIQA/scripts/run_agentevolver_snips.sh --variant urdu
bash IRTIQA/scripts/run_agentevolver_appworld.sh --variant codeswitched

bash IRTIQA/scripts/run_autogenesis_drop.sh --variant english
bash IRTIQA/scripts/run_autogenesis_snips.sh --variant urdu
bash IRTIQA/scripts/run_autogenesis_appworld.sh --variant codeswitched
bash IRTIQA/scripts/run_autogenesis_skillsbench.sh --variant english
```

Set framework-specific paths locally, for example:

- `TOOLR0_INTEGRATION=/path/to/Tool-R0/arr_integration`
- `AGENTEVOLVER_ROOT=/path/to/AgentEvolver`
- `AUTOGENESIS_ROOT=/path/to/Autogenesis`
- `MODEL_BASE_URL=http://localhost:8000/v1`
- `MODEL_NAME=<served-model-name>`

Machine-specific paths, logs, results, caches, checkpoints, and personal
annotation outputs are intentionally excluded.
