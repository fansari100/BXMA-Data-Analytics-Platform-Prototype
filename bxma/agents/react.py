"""
ReAct (Reason + Act) Agent Framework
====================================

Implements the ReAct pattern for autonomous financial agents that can
perceive, reason, and act upon market signals with recorded decision factors.

The ReAct loop:
1. Thought: Agent reasons about the current state
2. Action: Agent selects and executes a tool
3. Observation: Agent receives feedback from the environment
4. Repeat until goal achieved or max iterations

References:
- Yao et al. (2022): ReAct: Synergizing Reasoning and Acting in Language Models
- Anthropic (2025): Constitutional AI for Financial Applications

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, TypeVar, Generic
import numpy as np


class AgentRole(Enum):
    """Agent roles in the risk swarm."""
    ARCHITECT = auto()      # High-level orchestrator
    ANALYST = auto()        # Domain-specific analysis
    EXECUTOR = auto()       # Trade/hedge execution
    MONITOR = auto()        # Continuous surveillance
    JUDGE = auto()          # Compliance validator


@dataclass
class Thought:
    """
    An agent's reasoning step.
    
    Captures the internal deliberation before taking action.
    """
    content: str
    confidence: float  # 0-1 confidence score
    reasoning_type: str  # e.g., "deductive", "analogical", "abductive"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "confidence": self.confidence,
            "reasoning_type": self.reasoning_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Action:
    """
    An action taken by the agent.
    
    Represents a tool invocation with parameters.
    """
    tool_name: str
    parameters: dict[str, Any]
    rationale: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Observation:
    """
    Feedback from the environment after an action.
    """
    content: Any
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "content": str(self.content)[:1000],  # Truncate for logging
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentState:
    """
    Complete state of an agent at a point in time.
    """
    agent_id: str
    role: AgentRole
    goal: str
    
    # Trajectory
    thoughts: list[Thought] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    
    # Memory
    working_memory: dict[str, Any] = field(default_factory=dict)
    long_term_memory: list[dict] = field(default_factory=list)
    
    # Status
    iteration: int = 0
    completed: bool = False
    final_answer: str | None = None
    
    def get_trajectory(self) -> list[dict]:
        """Get interleaved thought-action-observation trajectory."""
        trajectory = []
        
        for i in range(len(self.thoughts)):
            trajectory.append({"type": "thought", **self.thoughts[i].to_dict()})
            if i < len(self.actions):
                trajectory.append({"type": "action", **self.actions[i].to_dict()})
            if i < len(self.observations):
                trajectory.append({"type": "observation", **self.observations[i].to_dict()})
        
        return trajectory
    
    def to_json(self) -> str:
        """Serialize state to JSON for audit logging."""
        return json.dumps({
            "agent_id": self.agent_id,
            "role": self.role.name,
            "goal": self.goal,
            "trajectory": self.get_trajectory(),
            "working_memory": {k: str(v)[:500] for k, v in self.working_memory.items()},
            "iteration": self.iteration,
            "completed": self.completed,
            "final_answer": self.final_answer,
        }, indent=2)


@dataclass
class AgentConfig:
    """Configuration for ReAct agents."""
    
    max_iterations: int = 10
    temperature: float = 0.7  # LLM temperature for reasoning
    
    # Tool configuration
    available_tools: list[str] = field(default_factory=list)
    tool_timeout_seconds: float = 30.0
    
    # Safety
    require_judge_approval: bool = True
    max_actions_per_minute: int = 10
    
    # Memory
    working_memory_size: int = 20
    include_long_term_memory: bool = True
    
    # Audit
    log_all_thoughts: bool = True
    explain_actions: bool = True


class FinancialTool(ABC):
    """
    Abstract base class for financial tools available to agents.
    
    Tools are the "hands" of the agent - they execute actions in the environment.
    """
    
    name: str
    description: str
    parameters_schema: dict[str, Any]
    
    @abstractmethod
    def execute(self, **kwargs) -> Observation:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        """Validate parameters before execution."""
        pass
    
    def __call__(self, **kwargs) -> Observation:
        """Callable interface for tool execution."""
        valid, error = self.validate_parameters(**kwargs)
        if not valid:
            return Observation(
                content=None,
                success=False,
                error_message=f"Invalid parameters: {error}",
            )
        
        try:
            return self.execute(**kwargs)
        except Exception as e:
            return Observation(
                content=None,
                success=False,
                error_message=f"Tool execution error: {str(e)}",
            )


class ReActAgent:
    """
    ReAct Agent implementation for autonomous risk management.
    
    The agent follows the Thought -> Action -> Observation loop
    to accomplish financial analysis and trading goals.
    """
    
    def __init__(
        self,
        role: AgentRole,
        config: AgentConfig,
        tools: dict[str, FinancialTool],
        llm_interface: Callable[[str], str] | None = None,
    ):
        self.role = role
        self.config = config
        self.tools = tools
        self.llm = llm_interface or self._default_llm
        
        self.agent_id = str(uuid.uuid4())[:8]
        self.state: AgentState | None = None
    
    def _default_llm(self, prompt: str) -> str:
        """
        Default LLM interface (placeholder for actual LLM integration).
        
        In production, this connects to FinGPT, Claude, or GPT-4.
        """
        # This is a placeholder - in production, call the actual LLM API
        return f"Analyzing: {prompt[:100]}..."
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""You are a {self.role.name} agent in a quantitative risk management system.

Your available tools:
{tools_desc}

Follow the ReAct pattern:
1. THOUGHT: Reason about what you need to do
2. ACTION: Select a tool and parameters
3. OBSERVATION: Process the result
4. Repeat until goal is achieved

Format your response as:
THOUGHT: <your reasoning>
ACTION: <tool_name>
PARAMETERS: <json parameters>

Or if you have the final answer:
THOUGHT: <final reasoning>
ANSWER: <your final response>

Be precise, quantitative, and risk-aware in your analysis.
"""
    
    def _build_step_prompt(self, goal: str) -> str:
        """Build prompt for current step."""
        trajectory_str = ""
        if self.state:
            for item in self.state.get_trajectory()[-10:]:  # Last 10 items
                if item["type"] == "thought":
                    trajectory_str += f"\nTHOUGHT: {item['content']}"
                elif item["type"] == "action":
                    trajectory_str += f"\nACTION: {item['tool_name']}"
                    trajectory_str += f"\nPARAMETERS: {json.dumps(item['parameters'])}"
                elif item["type"] == "observation":
                    trajectory_str += f"\nOBSERVATION: {item['content']}"
        
        return f"""Goal: {goal}

Previous trajectory:
{trajectory_str if trajectory_str else "(Starting fresh)"}

What is your next step?"""
    
    def _parse_response(self, response: str) -> tuple[Thought, Action | None, str | None]:
        """
        Parse LLM response into structured components.
        
        Returns:
            Tuple of (thought, action_or_none, final_answer_or_none)
        """
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|ANSWER:|$)", response, re.DOTALL)
        action_match = re.search(r"ACTION:\s*(\w+)", response)
        params_match = re.search(r"PARAMETERS:\s*(\{.+?\})", response, re.DOTALL)
        answer_match = re.search(r"ANSWER:\s*(.+)$", response, re.DOTALL)
        
        thought_content = thought_match.group(1).strip() if thought_match else "Continuing analysis..."
        thought = Thought(
            content=thought_content,
            confidence=0.8,  # Could be extracted from response
            reasoning_type="analytical",
        )
        
        action = None
        if action_match and params_match:
            tool_name = action_match.group(1)
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {}
            
            action = Action(
                tool_name=tool_name,
                parameters=params,
                rationale=thought_content,
            )
        
        final_answer = answer_match.group(1).strip() if answer_match else None
        
        return thought, action, final_answer
    
    def run(self, goal: str) -> AgentState:
        """
        Run the agent to achieve a goal.
        
        Args:
            goal: Natural language description of the goal
            
        Returns:
            Final agent state with trajectory and answer
        """
        self.state = AgentState(
            agent_id=self.agent_id,
            role=self.role,
            goal=goal,
        )
        
        system_prompt = self._build_system_prompt()
        
        for iteration in range(self.config.max_iterations):
            self.state.iteration = iteration
            
            # Build prompt for this step
            step_prompt = self._build_step_prompt(goal)
            full_prompt = f"{system_prompt}\n\n{step_prompt}"
            
            # Get LLM response
            response = self.llm(full_prompt)
            
            # Parse response
            thought, action, final_answer = self._parse_response(response)
            self.state.thoughts.append(thought)
            
            # Check for completion
            if final_answer:
                self.state.completed = True
                self.state.final_answer = final_answer
                break
            
            # Execute action
            if action:
                self.state.actions.append(action)
                
                if action.tool_name in self.tools:
                    tool = self.tools[action.tool_name]
                    observation = tool(**action.parameters)
                else:
                    observation = Observation(
                        content=None,
                        success=False,
                        error_message=f"Unknown tool: {action.tool_name}",
                    )
                
                self.state.observations.append(observation)
                
                # Update working memory
                self.state.working_memory[f"obs_{iteration}"] = observation.content
        
        return self.state
    
    async def run_async(self, goal: str) -> AgentState:
        """Async version of run for concurrent agent execution."""
        # In production, this would use async LLM calls
        return self.run(goal)


class AgentMemory:
    """
    Long-term memory system for agents.
    
    Stores past experiences and enables retrieval-augmented reasoning.
    """
    
    def __init__(self, max_memories: int = 1000):
        self.max_memories = max_memories
        self.memories: list[dict] = []
        self.embeddings: list[np.ndarray] = []
    
    def add(self, content: str, metadata: dict | None = None):
        """Add a memory."""
        memory = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        self.memories.append(memory)
        
        # Prune if needed
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search memories by relevance.
        
        In production, this uses vector similarity search.
        """
        # Simple keyword matching (placeholder for vector search)
        query_words = set(query.lower().split())
        
        scored = []
        for memory in self.memories:
            content_words = set(memory["content"].lower().split())
            overlap = len(query_words & content_words)
            scored.append((overlap, memory))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored[:top_k]]
    
    def get_recent(self, n: int = 10) -> list[dict]:
        """Get most recent memories."""
        return self.memories[-n:]


class ThoughtChain:
    """
    Chain of thought reasoning for complex multi-step problems.
    
    Implements explicit reasoning chains with verification.
    """
    
    def __init__(self):
        self.steps: list[Thought] = []
        self.verified: list[bool] = []
    
    def add_step(self, thought: Thought, verified: bool = False):
        """Add a reasoning step."""
        self.steps.append(thought)
        self.verified.append(verified)
    
    def get_chain(self) -> str:
        """Get the full reasoning chain as text."""
        lines = []
        for i, (step, verified) in enumerate(zip(self.steps, self.verified)):
            status = "✓" if verified else "?"
            lines.append(f"Step {i+1} [{status}]: {step.content}")
        return "\n".join(lines)
    
    def confidence_score(self) -> float:
        """Compute overall confidence from chain."""
        if not self.steps:
            return 0.0
        
        # Product of individual confidences
        confidences = [s.confidence for s in self.steps]
        return float(np.prod(confidences))
