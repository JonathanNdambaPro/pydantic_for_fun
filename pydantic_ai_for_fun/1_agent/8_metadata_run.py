from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

@dataclass
class Deps:
    tenant: str


agent = Agent[Deps](
    'openai:gpt-5.2',
    deps_type=Deps,
    metadata=lambda ctx: {'tenant': ctx.deps.tenant},  # agent-level metadata
)

result = agent.run_sync(
    'What is the capital of France?',
    deps=Deps(tenant='tenant-123'),
    metadata=lambda ctx: {'num_requests': ctx.usage.requests},  # per-run metadata
)
logger.info(result.output)
#> The capital of France is Paris.
logger.info(result.metadata)
#> {'tenant': 'tenant-123', 'num_requests': 1}