from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    PYDANTIC_AI_GATEWAY_API_KEY: str
    GOOGLE_API_KEY: str
    OPENROUTER_API_KEY: str
    LOGFIRE_TOKEN: str
    OPENWEATHERMAP_API_KEY: str


settings = Settings()
