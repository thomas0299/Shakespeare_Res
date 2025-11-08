#!/bin/bash

# --- Common Parameters ---
EPOCHS=10
DEEP_TYPES=("deep_ia" "deep_esn" "deep_esn_d" "grouped_esn")
NUM_RES_RANGE=$(seq 1 10) # 1 to 2

# --- Script-specific parameters ---
SCRIPTS_TO_RUN=("train_reservoir.py" "train_att_reservoir.py")
RUN_PARALLEL=false

# ==========================================================
if [ "$RUN_PARALLEL" = true ]; then
  echo "Starting all parallel jobs..."
else
  echo "Starting all sequential jobs..."
fi
# ==========================================================

# Loop over each script, each deep type, and each reservoir count
for SCRIPT_NAME in "${SCRIPTS_TO_RUN[@]}"; do
  for DEEP_TYPE in "${DEEP_TYPES[@]}"; do
    for i in $NUM_RES_RANGE; do
      
      # Build the common part of the command
      # We use an array for safety with arguments
      CMD_ARGS=(
        "--EPOCHS" "$EPOCHS"
        "--NUM_RES" "$i"
        "--DEEP_TYPE" "$DEEP_TYPE"
      )
      
      # --- Add script-specific arguments and define log file ---
      if [ "$SCRIPT_NAME" == "train_att_reservoir.py" ]; then
        LOG_FILE="run_att_reservoir_${DEEP_TYPE}_num${i}.log"
        JOB_DESC="att_reservoir $DEEP_TYPE NUM_RES=$i"
      
      elif [ "$SCRIPT_NAME" == "train_reservoir.py" ]; then
        # No extra arguments needed
        LOG_FILE="run_reservoir_${DEEP_TYPE}_num${i}.log"
        JOB_DESC="reservoir $DEEP_TYPE NUM_RES=$i"
      
      else
        echo "Skipping unknown script: $SCRIPT_NAME"
        continue
      fi
      # --- End of script-specific block ---

      echo "Starting job: $JOB_DESC. Logging to $LOG_FILE"

      if [ "$RUN_PARALLEL" = true ]; then
          # Run in PARALLEL (send to background with &)
          nohup python "$SCRIPT_NAME" "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
      else
          # Run SEQUENTIALLY (wait for it to finish, no &)
          # We still use nohup to correctly redirect output to the log file
          nohup python "$SCRIPT_NAME" "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1
          echo "Finished job: $JOB_DESC"
      fi
      
    done
  done
done

echo "All jobs are running in the background."
echo "Use 'jobs -l' to see them or 'ps aux | grep python' to check processes."