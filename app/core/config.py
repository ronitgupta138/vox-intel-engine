from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Presentation Intelligence Engine"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Multi-Modal AI & Digital Signal Processing Engine for Speech Delivery and Confidence Evaluation"
    
    # Target Speech Benchmark Standards
    IDEAL_WPM_MIN: float = 120.0
    IDEAL_WPM_MAX: float = 160.0
    IDEAL_PITCH_HZ_MALE_MIN: float = 85.0
    IDEAL_PITCH_HZ_MALE_MAX: float = 180.0
    IDEAL_PITCH_HZ_FEMALE_MIN: float = 165.0
    IDEAL_PITCH_HZ_FEMALE_MAX: float = 255.0
    MAX_FILLER_WORD_RATIO: float = 0.03  # Max 3% filler words recommended

    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
