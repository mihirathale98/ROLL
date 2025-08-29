from dataclasses import dataclass
from roll.agentic.env.base import BaseEnvConfig


@dataclass
class TauRetailEnvConfig(BaseEnvConfig):
    """Configuration for tau-bench retail environment."""
    
    # Tau-bench specific settings
    user_strategy: str = "llm"  # llm, react, verify, reflection
    user_model: str = "gpt-4o"
    user_provider: str = "openai"
    
    # Environment settings
    max_steps: int = 30
    env_instruction: str = "You are a customer service agent helping with retail orders. Use the available tools to assist customers with returns, exchanges, modifications, and cancellations."
    action_pattern: str = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    
    # Task selection
    task_subset: str = "all"  # all, returns, exchanges, modifications, cancellations
    randomize_tasks: bool = True
    mode: str = "train"  # train, val/eval - determines which task set to use