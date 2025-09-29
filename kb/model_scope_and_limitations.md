## Brief

- Clear statement of what the app's models can — and cannot — provide. Include expected domain and failure modes.

## Scope

- Models are trained on curated public & internal datasets for early triage: binary toxicity classifier, pathway-level (Tox21-type) predictors, ADMET regression models.

- Primary use: rapid prioritization and hypothesis generation for medicinal chemistry.

## Limitations & failure modes

- Domain of applicability: unusual chemotypes (large macrocycles, metal complexes, peptides) may be outside training distribution and produce unreliable predictions.

- Dose & exposure: models predict intrinsic properties or in vitro-like activity — they do not model dose-dependent effects or in vivo pharmacokinetics precisely.

- Metabolic context: predictions typically do not include full metabolite predictions or species-specific metabolism differences.

- Hallucination risk for LLMs: if an LLM is used to phrase answers, it should receive only KB text and explicit guardrails to prevent invented claims.