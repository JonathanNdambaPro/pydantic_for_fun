import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext

load_dotenv()

logfire.configure()
logfire.instrument_pydantic_ai()

logger.configure(handlers=[logfire.loguru_handler()])

roulette_agent = Agent(
    'gateway/openai:gpt-5.2',
    deps_type=int,
    output_type=bool,
    instructions=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


def main():
    success_number = 18
    result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
    logger.info(result.output)

    result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
    logger.info(result.output)


if __name__ == "__main__":
    main()
