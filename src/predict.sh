
#!/bin/bash
# export LAYERNORM_TYPE=fast_layernorm
# export USE_DEEPSPEED_EVO_ATTENTION=true
#### CUTLASS_PATH example
# export CUTLASS_PATH=../your/cutlass


PYTHON_FILE=./run_aurobind.py


INPUT_DATA=./examples/5S8I_A.yaml
OUTPUT_DIR=./output
SEED=42
NUM_DIFFUSION_SAMPLES=1
CACHE_DATA_DIR=./cache_data

python $PYTHON_FILE \
$INPUT_DATA \
--seed $SEED \
--out_dir $OUTPUT_DIR \
--num_diffusion_samples $NUM_DIFFUSION_SAMPLES \
--cache $CACHE_DATA_DIR



# # The following is a demo to use Accelerate to run the script on a single machine with multiple GPUs.
# INPUT_DATA=./examples
# OUTPUT_DIR=./output
# SEED=42,66
# NUM_DIFFUSION_SAMPLES=5
# CACHE_DATA_DIR=./cache_data

# accelerate launch \
# --multi_gpu \
# --num_processes 2 \
# --num_machines 1 \
# --main_process_port 20472 \
# $PYTHON_FILE \
# $INPUT_DATA \
# --seed $SEED \
# --out_dir $OUTPUT_DIR \
# --num_diffusion_samples $NUM_DIFFUSION_SAMPLES \
# --cache $CACHE_DATA_DIR


# # The following is a demo to use Accelerate with Config file to run the script on a single machine with multiple GPUs.
# ## You can modify the config file to set the number of GPUs or number of Machines and other parameters.
# ACCELERATE_CONFIG_FILE=./accelerator_single_machine.json
# INPUT_DATA=./examples
# OUTPUT_DIR=./output
# SEED=42,66
# # SEED=42,66,88,101,2025
# NUM_DIFFUSION_SAMPLES=5
# CACHE_DATA_DIR=./cache_data

# accelerate launch \
# --config_file $ACCELERATE_CONFIG_FILE \
# $PYTHON_FILE \
# $INPUT_DATA \
# --seed $SEED \
# --out_dir $OUTPUT_DIR \
# --num_diffusion_samples $NUM_DIFFUSION_SAMPLES \
# --cache $CACHE_DATA_DIR


