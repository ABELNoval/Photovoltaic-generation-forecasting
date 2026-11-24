import pytest

from src.data import weather


def test_fetch_weather_rejects_duplicate_timestamps(monkeypatch):
    duplicate_time = "2014-01-01T00:00"
    payload = {
        "hourly": {
            "time": [duplicate_time, duplicate_time],
            **{variable: [1.0, 2.0] for variable in weather.VARIABLES},
        },
        "latitude": 52.1,
        "longitude": 5.1,
    }
    monkeypatch.setattr(weather, "_fetch_period", lambda *args: payload)

    with pytest.raises(ValueError, match=r"1 timestamps duplicados.*2014-01-01T00:00:00"):
        weather.fetch_weather(52.1, 5.1, "2014-01-01", "2014-01-01")
