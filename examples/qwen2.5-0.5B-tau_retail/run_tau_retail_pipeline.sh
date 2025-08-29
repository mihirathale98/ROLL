#!/bin/bash

# Run tau-bench retail environment training with ROLL
# Make sure you have:
# 1. tau-bench in the tau-bench/ directory
# 2. OpenAI API key set: export OPENAI_API_KEY=your_key_here

echo "Starting tau-bench retail training with ROLL..."

# Set environment variables if not already set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. User simulator will not work."
    echo "Please run: export OPENAI_API_KEY=your_key_here"
fi

# Run the agentic pipeline with tau-retail environment
python ../../examples/start_agentic_pipeline.py \
    --config-name tau_retail_config \
    --config-path examples/qwen2.5-0.5B-tau_retail \
    rollout_batch_size=8 \
    max_steps=200 \
    logging_steps=5 \
    eval_steps=25