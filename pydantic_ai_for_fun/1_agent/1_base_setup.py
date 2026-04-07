import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

logfire.configure(handlers=[logfire.loguru_handler()])
logfire.instrument_pydantic_ai()
logfire.instrument_sqlite3()

load_dotenv()

agent = Agent   (
    # "openrouter:anthropic/claude-sonnet-4-5",
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Be concise, reply with one sentence.',
    capabilities=[Thinking(), WebSearch()]
)

result = agent.run_sync('Where does "hello world" come from?')

logger.info(result.output)
