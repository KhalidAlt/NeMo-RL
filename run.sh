RESTORE_FRON_PATH=$1
TRAIN_FILE=$2
VALID_FILE=$3
EXP_PATH=$4
NUM_NODES=$5
EXP_NAME=$6
MBS=$7
BATCH_SIZE=$8
TP=$9
PP=${10}
LR_RATE=${11}
CKPT_INTERVAL=${12}
EPOCH=${13}
checkpoint_type=${14}
NUM_WORKER_FOR_DATALOADER=1


if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
    echo "=== LOCAL_RANK 0: Setting up Ray cluster for node ${NODE_RANK:-0} ==="

    # Copy NeMo-RL files only once per node
    # cp -r /opt/nemo-rl/* .
    # Build the training command
    COMMAND=$(cat << 'EOF'
 uv run run_grpo_math.py \
    --config=conf/grpo.yaml \
    cluster.num_nodes=$NUM_NODES \
    policy.train_micro_batch_size=$MBS \
    policy.train_global_batch_size=$BATCH_SIZE \
    policy.dtensor_cfg.tensor_parallel_size=$TP \
    policy.dtensor_cfg.pipeline_parallel_size=$PP \
    checkpointing.checkpoint_dir="$EXP_PATH" \
    logger.wandb_enabled=True \
    logger.wandb.name="$EXP_NAME"
EOF
)
    
    # Export variables for the multinode script
    export RESTORE_FROM_PATH TRAIN_FILE VALID_FILE EXP_PATH EXP_NAME
    export MBS BATCH_SIZE TP PP LR_RATE CKPT_INTERVAL EPOCH
    
    # Run the multinode Ray setup
    bash run_multinode.sh "$COMMAND" "$NUM_NODES"
    
else
    echo "=== LOCAL_RANK ${LOCAL_RANK}: Waiting for Ray setup on this node ==="
    
    # Wait for Ray to be set up by LOCAL_RANK 0
    while ! ray status > /dev/null 2>&1; do
        echo "Waiting for Ray to be ready on node ${NODE_RANK}..."
        sleep 5
    done
    
    echo "Ray is ready on node ${NODE_RANK}, process ${LOCAL_RANK} can proceed"
    
    # For non-zero LOCAL_RANK processes, just wait for completion
    # The actual training is handled by the Ray cluster
    while [ ! -f /tmp/training_complete ]; do
        sleep 30
        # Check if Ray is still running
        if ! ray status > /dev/null 2>&1; then
            echo "Ray cluster stopped, exiting..."
            break
        fi
    done
fi

echo "Process LOCAL_RANK=${LOCAL_RANK} on NODE_RANK=${NODE_RANK} completed"
