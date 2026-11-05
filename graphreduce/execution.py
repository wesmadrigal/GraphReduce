"""Results from full or bounded graph execution."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class ExecutionResult:
    """An operation's output: a dataframe, or a SQL table/view reference.

    ``do_reduce`` returns the reduction before it is joined into its parent.
    ``edge`` identifies the relationship for edge-level operations. Results
    refer to live graph data; they are not deep copies or resume checkpoints.
    """

    node: Any
    operation: str
    data: Any
    edge: Optional[Tuple[Any, Any]] = None
