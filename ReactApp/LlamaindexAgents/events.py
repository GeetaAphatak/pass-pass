"""
Event definitions for the LlamaIndex agent workflow.
"""
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.llms import ChatMessage
from dataclasses import dataclass
from typing import List

@dataclass
class PrepEvent:
    """Event for preparation phase."""
    pass

@dataclass
class InputEvent:
    """Event for handling input."""
    input: List[ChatMessage]

@dataclass
class StreamEvent:
    """Event for handling stream data."""
    delta: str

@dataclass
class ToolCallEvent:
    """Event for tool calls."""
    tool_calls: List[ToolSelection]

@dataclass
class FunctionOutputEvent:
    """Event for function output."""
    output: ToolOutput
