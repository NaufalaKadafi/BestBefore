"""BestBefore: freshness labels for AI answers.

Pipeline: extract time-sensitive claims (LLM via Bifrost) -> gather live evidence
(Tavily) -> judge the evidence (LLM via Bifrost) -> label with rules in code.
"""

__version__ = "0.1.0"
