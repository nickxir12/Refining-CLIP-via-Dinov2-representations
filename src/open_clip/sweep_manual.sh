#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd -P)"

# Load defaults (LR etc. stay fixed from hparams.sh)
set -a
source "${SCRIPT_DIR}/hparams.sh"
source "${SCRIPT_DIR}/cluster_env.sh"
set +a

ORIG_RUN_NAME="${RUN_NAME:-Dino_variant_$(date +%Y%m%d-%H%M%S)}"
: "${RUN_GROUP:=flickr30k_dino_manual_$(date +%Y%m%d-%H%M)}"
: "${DRY_RUN:=0}"

# ----------------------------------------------------
# Define runs explicitly — only DINO hyperparams vary
# ----------------------------------------------------
RUNS=(


# Example 1: geometry on

 "USE_DINO=0 USE_PROJECTION=False PROJECTION_TYPE=mlp LOSS_MODE=clip LAMBDA_ORIGINAL=1.0 \
 LAMBDA_SOFT=0.25 SOFT_MODE=kl_teacher SOFT_DINO_TO_TEXT=1 \
 TEXT_LAMBDA=0.5 TEXT_STUDENT_TEMP=0.02 \
 LAMBDA_WEIGHTED=0.0 RHO=0.1 C_CLIP=1.0 WEIGHT_TEXT_SYMMETRY=0 \
 LAMBDA_GEOM=0.0"

"USE_DINO=1 USE_PROJECTION=True PROJECTION_TYPE=mlp LOSS_MODE=clip LAMBDA_ORIGINAL=1.0 \
 LAMBDA_SOFT=0.5 SOFT_MODE=kl_teacher SOFT_DINO_TO_TEXT=1 \
 TEXT_LAMBDA=0.5 TEXT_STUDENT_TEMP=0.02 \
 LAMBDA_WEIGHTED=0.0 RHO=0.1 C_CLIP=1.0 WEIGHT_TEXT_SYMMETRY=0 \
 LAMBDA_GEOM=0.0"

"USE_DINO=1 USE_PROJECTION=True PROJECTION_TYPE=mlp LOSS_MODE=clip LAMBDA_ORIGINAL=1.0 \
 LAMBDA_SOFT=0.5 SOFT_MODE=kl_teacher SOFT_DINO_TO_TEXT=0 \
 TEXT_LAMBDA=0.5 TEXT_STUDENT_TEMP=0.02 \
 LAMBDA_WEIGHTED=0.0 RHO=0.1 C_CLIP=1.0 WEIGHT_TEXT_SYMMETRY=0 \
 LAMBDA_GEOM=0.0"

"USE_DINO=1 USE_PROJECTION=True PROJECTION_TYPE=linear LOSS_MODE=clip LAMBDA_ORIGINAL=1.0 \
 LAMBDA_SOFT=0.5 SOFT_MODE=kl_teacher SOFT_DINO_TO_TEXT=1 \
 TEXT_LAMBDA=0.5 TEXT_STUDENT_TEMP=0.02 \
 LAMBDA_WEIGHTED=0.0 RHO=0.1 C_CLIP=1.0 WEIGHT_TEXT_SYMMETRY=0 \
 LAMBDA_GEOM=0.0"


)



)
# ----------------------------------------------------

for RUNCONF in "${RUNS[@]}"; do
  eval "export $RUNCONF"   # set these vars for this run

  # Make suffix with ONLY the DINO hyperparams
  SUF="dino_lorig=${LAMBDA_ORIGINAL}_lsoft=${LAMBDA_SOFT}_softm=${SOFT_MODE}_sdtt=${SOFT_DINO_TO_TEXT}_tl=${TEXT_LAMBDA}_tst=${TEXT_STUDENT_TEMP}_lW=${LAMBDA_WEIGHTED}_rho=${RHO}_cclip=${C_CLIP}_wts=${WEIGHT_TEXT_SYMMETRY}"
  SUF="${SUF//./p}"   # dots → p
  SUF="${SUF// /_}"   # spaces → underscores

  export RUN_NAME="${ORIG_RUN_NAME}_${RUN_GROUP}_${SUF}"

  echo "[sweep] RUN_NAME=${RUN_NAME}"
  echo "        config: $RUNCONF"

  if [[ "$DRY_RUN" == "0" ]]; then
    jid="$("${SCRIPT_DIR}/sumbit.sh")"
    echo "[sweep] submitted job $jid"
  else
    echo "[sweep] DRY_RUN=1 -> not submitting"
  fi
done