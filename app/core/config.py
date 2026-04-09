from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    APP_NAME : str = "Carter.ai"
    ENVIORNMENT : str = "development"
    ANTHROPIC_API_KEY : str = ""
    MAX_UPLOAD_SIZE_MB : int = 50
    MIN_ROWS_REQUIRED : int = 100
    ALLOWED_EXTENSIONS : list[str] = [".csv", ".xlsx"]
    model_config = SettingsConfigDict(
        env_file = ".env",
        extra = "ignore"
    )
settings = Settings()