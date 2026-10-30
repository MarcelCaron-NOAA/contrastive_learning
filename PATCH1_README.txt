PATCH 1 — REPRODUCIBILITY FOUNDATION
====================================

Purpose
-------
This patch changes repository/configuration structure only. It does not alter
SYN-3, SYN-8, NFV training objectives, transforms, metrics, checkpoints, or
scientific evaluation logic.

What it changes
---------------
1. Replaces hard-coded personal paths in load_env.sh with one local root-level
   site_config.sh (ignored by Git) created from site_config.example.sh.
2. Replaces the machine dump environment.yml with a small URSA-oriented
   supported environment definition.
3. Preserves the historical environment/package dumps as environment-lock.yml
   and requirements-lock.txt for reference.
4. Rewrites README.md around SYN-3 / SYN-8 / NFV and the four primary
   diagnostic suites, including feature-space inversion.
5. Adds docs/DATA.md and docs/REPRODUCING_RESULTS.md.
6. Adds docs/REPOSITORY_CLEANUP.md, a file-by-file classification of current
   scripts and Slurm launchers as CANONICAL, SUPPORT, or HISTORY.
7. Expands .gitignore for local config/data/generated products.

Apply
-----
From the refactor/reproducible-v1 branch with a clean working tree:

  git status
  git apply --check /path/to/0001-reproducibility-foundation.patch
  git apply /path/to/0001-reproducibility-foundation.patch

Review:

  git status
  git diff --check
  git diff

Local URSA setup after applying
-------------------------------
Create the one local configuration file:

  cp site_config.example.sh site_config.sh

For Marcel's existing project, edit site_config.sh so the values point to the
existing working area/environment. PROJECT_DIR is detected automatically.

Then test the bootstrap:

  source load_env.sh

The existing environment can continue to be used for now. Do NOT recreate or
delete it just to test this patch. We will validate the smaller environment.yml
separately before making it the only supported environment.

Suggested commit
----------------

  git add -A
  git commit -m "Establish reproducible project configuration and documentation"

Validation performed before delivery
------------------------------------
- bash -n load_env.sh
- bash -n site_config.example.sh
- YAML parse of environment.yml
- YAML parse of environment-lock.yml
- git diff --cached --check
- verified load_env.sh fails clearly when site_config.sh is missing

Next patch
----------
Training consolidation:
- shared SYN trainer for SYN-3 and SYN-8
- tracked SYN-3/SYN-8 configs
- clean NFV naming/entry point without changing its objective
- canonical Slurm wrappers sourcing the central environment
