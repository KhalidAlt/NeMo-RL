#!/bin/bash

set -eoux pipefail

# Parse arguments
COMMAND=$1
NUM_NODES=$2

# Ray cluster configuration for Azure ML
RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT:-10001}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

# Azure ML provides these environment variables for distributed training
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6379}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
LOCAL_RANK=${LOCAL_RANK:-0}

# Calculate actual number of nodes from WORLD_SIZE and process_count_per_instance
PROCESSES_PER_NODE=8  # This is set in your Azure ML config
ACTUAL_NUM_NODES=$((WORLD_SIZE / PROCESSES_PER_NODE))

echo "=== Azure ML Distributed Training Info ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "NUM_NODES (requested): $NUM_NODES"
echo "ACTUAL_NUM_NODES (calculated): $ACTUAL_NUM_NODES"
echo "COMMAND: $COMMAND"
echo "=========================================="

# Function to start Ray head node
start_ray_head() {
    echo "Starting Ray head node on $MASTER_ADDR"
    
    ray start --head \
        --disable-usage-stats \
        --node-ip-address="$MASTER_ADDR" \
        --port=$RAY_HEAD_PORT \
        --ray-client-server-port=$RAY_CLIENT_SERVER_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --num-cpus=0 \
        --num-gpus=0 \
        --block &
    
    RAY_HEAD_PID=$!
    echo "Ray head started with PID: $RAY_HEAD_PID"
    
    # Wait for Ray head to be ready
    sleep 10
    
    # Verify Ray head is running
    timeout=60
    counter=0
    while ! ray status > /dev/null 2>&1; do
        if [ $counter -ge $timeout ]; then
            echo "ERROR: Ray head failed to start within $timeout seconds"
            exit 1
        fi
        echo "Waiting for Ray head to be ready... ($counter/$timeout)"
        sleep 1
        ((counter++))
    done
    
    echo "Ray head is ready!"
}

# Function to start Ray worker node
start_ray_worker() {
    local head_address="$MASTER_ADDR:$RAY_HEAD_PORT"
    echo "Starting Ray worker, connecting to head at $head_address"
    
    # Calculate resources per node (8 GPUs per Azure ML node)
    local gpus_per_node=8
    
    ray start --address="$head_address" \
        --disable-usage-stats \
        --resources="{\"worker_units\": $gpus_per_node}" \
        --block &
    
    RAY_WORKER_PID=$!
    echo "Ray worker started with PID: $RAY_WORKER_PID"
    
    # Wait for worker to connect
    sleep 5
}

# Function to wait for all workers to connect
wait_for_workers() {
    local expected_workers=$ACTUAL_NUM_NODES
    local expected_worker_units=$((8 * expected_workers))  # 8 worker_units per node
    
    echo "Waiting for $expected_workers worker nodes ($expected_worker_units worker_units) to connect..."
    
    timeout=600  # 10 minutes timeout for large clusters
    counter=0
    
    while [ $counter -lt $timeout ]; do
        # Get Ray cluster status
        if ray_status=$(ray status 2>/dev/null); then
            echo "=== Ray Status Check (attempt $((counter/5 + 1))) ==="
            
            # Extract total worker_units from the resources section
            # Look for line like "0.0/16.0 worker_units"
            if worker_units_line=$(echo "$ray_status" | grep "worker_units"); then
                echo "Worker units line: $worker_units_line"
                # Extract the total (second number) from "0.0/16.0 worker_units"
                total_worker_units=$(echo "$worker_units_line" | awk '{print $1}' | cut -d'/' -f2)
                
                # Convert to integer for comparison (remove decimal)
                total_worker_units_int=$(echo "$total_worker_units" | cut -d'.' -f1)
                
                echo "Expected worker_units: $expected_worker_units, Available: $total_worker_units_int"
                
                if [ "$total_worker_units_int" -ge "$expected_worker_units" ]; then
                    echo "✅ All workers connected! ($total_worker_units_int >= $expected_worker_units worker_units)"
                    echo ""
                    echo "Final cluster status:"
                    ray status
                    return 0
                fi
            else
                echo "worker_units not found in Ray status yet..."
            fi
            
            # Also check node count as backup
            active_nodes=$(echo "$ray_status" | grep -A 20 "Node status" | grep -c "node_" || echo "0")
            expected_total_nodes=$((expected_workers + 1))  # +1 for head node
            echo "Active nodes: $active_nodes (expected: $expected_total_nodes including head)"
            
        else
            echo "Ray cluster not ready yet..."
        fi
        
        sleep 5
        ((counter+=5))
    done
    
    echo "❌ ERROR: Not all workers connected within $timeout seconds"
    echo "Final Ray status:"
    ray status || true
    exit 1
}


# Function to cleanup Ray processes
cleanup_ray() {
    echo "Cleaning up Ray processes..."
    ray stop --force || true
    pkill -f "ray start" || true
}

# Set up cleanup trap
trap cleanup_ray EXIT

# Wait for other nodes to start (staggered startup)
sleep_time=$((NODE_RANK * 10))
echo "Node $NODE_RANK: Waiting $sleep_time seconds for staggered startup..."
sleep $sleep_time

# Main execution logic based on node role
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Node $NODE_RANK: Starting as Ray head node ==="
    
    # Start Ray head node
    start_ray_head
    
    # Also start worker on head node for compute
    sleep 5
    start_ray_worker
    
    # Wait for all workers to connect
    wait_for_workers
    
    # Start the training job once cluster is ready
    echo "=== Starting NeMo-RL GRPO training ==="
    
    # Create Ray cluster configuration for NeMo-RL
    # export RAY_ADDRESS="ray://localhost:$RAY_CLIENT_SERVER_PORT"
    
    # Substitute variables in the command
    EXPANDED_COMMAND=$(eval echo "\"$COMMAND\"")
    echo "Executing: $EXPANDED_COMMAND"
    
    # Run the training command
    eval "$EXPANDED_COMMAND"
    
    training_exit_code=$?
    
    echo "Training completed with exit code: $training_exit_code"
    
    # Signal other nodes to shutdown
    touch /tmp/training_complete
    
    exit $training_exit_code

else
    echo "=== Node $NODE_RANK: Starting as Ray worker node ==="
    
    # Wait longer for head node to start
    sleep 30
    
    # Start Ray worker
    start_ray_worker
    
    # Keep worker running until training is complete
    # Monitor for completion signal or process termination
    while true; do
        if [ -f /tmp/training_complete ]; then
            echo "Training completed, shutting down worker..."
            break
        fi
        
        # Check if Ray processes are still running
        if ! pgrep -f "ray start" > /dev/null; then
            echo "Ray processes stopped, exiting..."
            break
        fi
        
        sleep 30
    done
fi

echo "Node $NODE_RANK: Execution completed"
