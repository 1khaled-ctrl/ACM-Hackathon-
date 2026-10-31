"""Central config: Bedrock model/region and the agent's hard limits.

Bedrock is only reachable from inside the SageMaker environment — nothing
here can be exercised locally except by reading these constants. --dry-run
in run_gotrain.py skips model construction entirely so it still works on a
laptop with no AWS access.
"""

BEDROCK_MODEL_ID = "eu.anthropic.claude-sonnet-5-5"
BEDROCK_REGION = "eu-central-1"

DEFAULT_MAX_RUNS = 6
DEFAULT_RUN_MINUTES = 5
DEFAULT_MAX_TOOL_CALLS = 40
