# Tau-Bench Retail Training with ROLL

This example demonstrates training an LLM on customer service conversations using tau-bench's retail environment integrated with ROLL's agentic pipeline.

## Overview

The tau-bench retail environment simulates realistic customer service scenarios where:
- A user simulator acts as a customer with specific requests (returns, exchanges, modifications)
- The agent (LLM being trained) must use tools to help the customer
- Tasks are drawn from 200+ training scenarios covering various retail operations

## Setup

1. **Install dependencies**:
   ```bash
   # Install tau-bench (already included as submodule)
   cd tau-bench && pip install -e .
   
   # Set API key for user simulator
   export OPENAI_API_KEY=your_key_here
   ```

2. **Available tools**:
   - `get_user_details`: Get customer information
   - `get_order_details`: Get order information  
   - `cancel_pending_order`: Cancel orders
   - `modify_pending_order_*`: Modify orders (items, payment, address)
   - `return_delivered_order_items`: Process returns
   - `exchange_delivered_order_items`: Process exchanges

## Training

Run the training script:
```bash
./run_tau_retail_pipeline.sh
```

Or manually:
```bash
python ../../examples/start_agentic_pipeline.py \
    --config-name tau_retail_config \
    --config-path examples/qwen2.5-0.5B-tau_retail
```

## Configuration

Key parameters in `tau_retail_config.yaml`:
- `user_strategy`: User simulator type (llm, react, verify, reflection)
- `user_model`: Model for user simulation (default: gpt-4o)
- `max_steps`: Maximum conversation turns per episode
- `task_subset`: Which types of tasks to include (all, returns, exchanges, etc.)

## Training Details

- **Environment**: Tau-bench retail customer service scenarios
- **Algorithm**: PPO with KL penalty  
- **Reward**: Binary task completion (1.0 for success, 0.0 for failure)
- **Episode length**: Variable, based on conversation completion
- **Training data**: 200+ diverse customer service scenarios

## Example Conversation

```
User: Hi, I need to return my Electric Kettle and Wall Clock from order #W6067464
Agent: I'd be happy to help you with that return. Let me look up your order details first.
[Agent uses get_order_details tool]
Agent: I can see your order. To process the return, I'll need to use your credit card ending in 4190576. Let me process that return for you.
[Agent uses return_delivered_order_items tool]
Agent: Perfect! I've successfully processed the return for your Electric Kettle and Wall Clock. You should see the refund on your credit card within 5-7 business days.
User: ###STOP###
```

## Monitoring

- Training logs: `./logs/tau_retail/`
- Checkpoints: `./checkpoints/tau_retail/`
- Metrics: Conversation success rate, tool usage accuracy, response quality