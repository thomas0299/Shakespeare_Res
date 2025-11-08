#!/bin/bash

# --- Common Parameters ---
EPOCHS=10
LAYERS_RANGE=$(seq 1 10) # 1 to 10 (Analogous to NUM_RES_RANGE)


# --- Script-specific parameters ---
SCRIPT_TO_RUN="train_transformer.py"
RUN_PARALLEL=false

# ==========================================================
if [ "$RUN_PARALLEL" = true ]; then
  echo "Starting all parallel jobs..."
else
  echo "Starting all sequential jobs..."
fi
# ==========================================================

# Loop over each layer count
for i in $LAYERS_RANGE; do
  
  # Build the common part of the command
  # We use an array for safety with arguments
  CMD_ARGS=(
    "--EPOCHS" "$EPOCHS"
    "--LAYERS" "$i"
  )
  
  # --- Define log file and job description ---
  # Simplified log file name since we removed ARCH_TYPE
  LOG_FILE="run_transformer_layers${i}.log"
  JOB_DESC="transformer LAYERS=$i"
  # --- End of definitions ---

  echo "Starting job: $JOB_DESC. Logging to $LOG_FILE"

  if [ "$RUN_PARALLEL" = true ]; then
      # Run in PARALLEL (send to background with &)
      nohup python "$SCRIPT_TO_RUN" "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
  else
      # Run SEQUENTIALLY (wait for it to finish, no &)
      # We still use nohup to correctly redirect output to the log file
      nohup python "$SCRIPT_TO_RUN" "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1
      echo "Finished job: $JOB_DESC"
  fi
  
done

if [ "$RUN_PARALLEL" = true ]; then
  echo "All jobs are running in the background."
  echo "Use 'jobs -l' to see them or 'ps aux | grep python' to check processes."
else
  echo "All sequential jobs have completed."
fi