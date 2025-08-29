import json
import re
import sys
import os
from typing import Tuple, Dict, Any

from roll.agentic.env.base import BaseEnv
from roll.agentic.env.tau_retail.config import TauRetailEnvConfig

# Add tau-bench to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tau_bench_path = os.path.join(current_dir, "../../../../tau-bench")
sys.path.insert(0, tau_bench_path)

try:
    from tau_bench.envs.retail.env import RetailEnv
    from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
    from tau_bench.envs.retail.tasks_dev import TASKS_DEV
    from tau_bench.types import Action, RESPOND_ACTION_NAME
except ImportError as e:
    raise ImportError(f"Could not import tau-bench modules: {e}. Make sure tau-bench is in the correct path.")


class TauRetailEnv(BaseEnv):
    """
    Adapter that wraps tau-bench retail environment for ROLL's agentic training.
    Bridges tau-bench's tool-calling interface with ROLL's text-based interface.
    """
    
    def __init__(self, config=None, mode="train", **kwargs):
        self.config = config or TauRetailEnvConfig()
        BaseEnv.__init__(self, config=self.config)
        
        # Initialize tau-bench retail environment
        self.tau_env = RetailEnv(
            user_strategy=self.config.user_strategy,
            user_model=self.config.user_model,
            user_provider=self.config.user_provider
        )
        
        # Select task set based on mode
        self.mode = mode
        if mode == "train":
            self.tasks = TASKS_TRAIN
        elif mode == "val" or mode == "eval":
            self.tasks = TASKS_DEV
        else:
            # Default to train tasks
            self.tasks = TASKS_TRAIN
            
        self.current_task_index = None
        self.conversation_history = []
        
    def reset(self, seed=None, task_index=None, **kwargs) -> Tuple[str, dict]:
        """Reset environment with a random or specified task."""
        if task_index is None:
            task_index = seed % len(self.tasks) if seed is not None else None
            
        env_reset_response = self.tau_env.reset(task_index=task_index)
        self.current_task_index = self.tau_env.task_index
        self.conversation_history = []
        
        # Initial observation is the user's first message
        observation = env_reset_response.observation
        self.conversation_history.append(f"User: {observation}")
        
        info = {
            "task_index": self.current_task_index,
            "task_instruction": self.tau_env.task.instruction,
            "conversation_turn": 0
        }
        
        return observation, info
        
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Parse action from LLM text output
        parsed_action = self.parse_action(action)
        
        # Execute action in tau-bench environment
        env_response = self.tau_env.step(parsed_action)
        
        # Update conversation history
        if parsed_action.name == RESPOND_ACTION_NAME:
            self.conversation_history.append(f"Agent: {parsed_action.kwargs['content']}")
        else:
            self.conversation_history.append(f"Agent used tool: {parsed_action.name}")
            
        self.conversation_history.append(f"User: {env_response.observation}")
        
        # Convert to ROLL format
        observation = env_response.observation
        reward = float(env_response.reward) if env_response.done else 0.0
        terminated = env_response.done
        truncated = False
        
        info = {
            "conversation_turn": len(self.conversation_history) // 2,
            "action_taken": parsed_action.name,
            "source": env_response.info.source if hasattr(env_response.info, 'source') else 'unknown'
        }
        
        if terminated and hasattr(env_response.info, 'reward_info'):
            info["reward_info"] = env_response.info.reward_info
            
        return observation, reward, terminated, truncated, info
        
    def parse_action(self, text: str) -> Action:
        """Parse LLM text output into tau-bench Action format."""
        # Try to extract tool call from structured format
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        if tool_call_match:
            try:
                tool_data = json.loads(tool_call_match.group(1))
                return Action(
                    name=tool_data.get("name", "respond"),
                    kwargs=tool_data.get("kwargs", {"content": text})
                )
            except json.JSONDecodeError:
                pass
        
        # Try to extract function call pattern
        func_match = re.search(r'(\w+)\((.*?)\)', text)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            
            # Parse arguments
            kwargs = {}
            if args_str.strip():
                try:
                    # Simple parsing for key=value pairs
                    for arg in args_str.split(','):
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            key = key.strip().strip('"\'')
                            value = value.strip().strip('"\'')
                            kwargs[key] = value
                except:
                    pass
                    
            return Action(name=func_name, kwargs=kwargs)
        
        # Default to respond action
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": text})
        
    def render(self, mode: str = "text") -> str:
        """Render the current state of the environment."""
        if mode == "text":
            return "\n".join(self.conversation_history[-6:])  # Last 3 turns
        return str(self.conversation_history)
        
    def get_all_actions(self):
        """Get list of available actions."""
        actions = [RESPOND_ACTION_NAME]
        if hasattr(self.tau_env, 'tools_map'):
            actions.extend(self.tau_env.tools_map.keys())
        return actions
        
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current task."""
        if self.current_task_index is not None:
            task = self.tasks[self.current_task_index]
            return {
                "task_index": self.current_task_index,
                "instruction": task.instruction,
                "user_id": task.user_id,
                "expected_actions": [action.name for action in task.actions]
            }
        return {}