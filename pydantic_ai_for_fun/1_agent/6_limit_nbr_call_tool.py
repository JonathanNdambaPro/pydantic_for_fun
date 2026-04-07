import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

agent = Agent('gateway/anthropic:claude-sonnet-4-6')

# @tool_plain : bypass l'injection du RunContext (utile pour des fonctions purement utilitaires)
@agent.tool_plain
def do_work() -> str:
    return 'ok'

try:
    # usage_limits agit comme un circuit breaker
    agent.run_sync('Please call the tool twice', usage_limits=UsageLimits(tool_calls_limit=1))
except UsageLimitExceeded as e:
    # Triggered dès que l'agent tente le 2ème call
    logger.warning(f"Circuit breaker activé : {e}")
