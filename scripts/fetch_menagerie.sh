#!/usr/bin/env bash
# Fetch the DeepMind MuJoCo Menagerie models the sim uses (git-ignored, ~30 MB).
set -euo pipefail
cd "$(dirname "$0")/../sim"
[ -d menagerie ] || git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git menagerie
cd menagerie && git sparse-checkout set shadow_hand franka_emika_panda
echo "menagerie ready: $(pwd)"
