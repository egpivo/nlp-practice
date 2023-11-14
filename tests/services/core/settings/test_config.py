from nlp_practice.service.core.settings.config import Settings


# Test default values
def test_default_values():
    settings = Settings()
    assert settings.app_name == "LLM Services"


# Test setting values
def test_setting_values():
    app_name_value = "Test App Name"
    settings = Settings(app_name=app_name_value)
    assert settings.app_name == app_name_value
