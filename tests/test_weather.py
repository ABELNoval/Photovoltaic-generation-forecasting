import json
from argparse import Namespace

import pytest
import requests

from src.data import weather


@pytest.mark.parametrize("error_type", [requests.Timeout, requests.HTTPError])
def test_fetch_weather_reports_year_for_request_errors(monkeypatch, error_type):
    def fail_request(*args):
        raise error_type("request failed")

    monkeypatch.setattr(weather, "_fetch_period", fail_request)

    with pytest.raises(RuntimeError, match=r"2014 \(2014-01-01 a 2014-12-31\)") as exc_info:
        weather.fetch_weather(52.1, 5.1, "2014-01-01", "2014-12-31")

    assert isinstance(exc_info.value.__cause__, error_type)


def test_fetch_period_requests_land_grid_cell(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def fake_get(url, params, timeout):
        request.update(params=params, timeout=timeout)
        return Response()

    monkeypatch.setattr(weather.requests, "get", fake_get)

    weather._fetch_period(52.1, 5.1, "2014-01-01", "2014-12-31")

    assert request["params"]["cell_selection"] == "land"


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


def test_command_saves_requested_and_grid_coordinates(monkeypatch, tmp_path):
    payload = {
        "hourly": {
            "time": ["2014-01-01T00:00"],
            **{variable: [1.0] for variable in weather.VARIABLES},
        },
        "latitude": 52.125,
        "longitude": 5.125,
    }
    monkeypatch.setattr(weather, "_fetch_period", lambda *args: payload)
    monkeypatch.setattr(weather, "RAW_DIR", tmp_path)
    args = Namespace(
        system_id=None,
        lat=52.10875,
        lon=5.125278,
        name="ID003",
        start="2014-01-01",
        end="2014-01-01",
    )

    weather._cmd(args)

    metadata_path = tmp_path / "weather_ID003.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["requested_coordinates"] == {
        "latitude": 52.10875,
        "longitude": 5.125278,
    }
    assert metadata["grid_coordinates"] == {
        "latitude": 52.125,
        "longitude": 5.125,
    }
    assert metadata["cell_selection"] == "land"
    assert metadata["period"] == {"start": "2014-01-01", "end": "2014-01-01"}


def test_command_preserves_previous_outputs_when_later_year_fails(monkeypatch, tmp_path):
    payload = {
        "hourly": {
            "time": ["2014-01-01T00:00"],
            **{variable: [1.0] for variable in weather.VARIABLES},
        },
        "latitude": 52.125,
        "longitude": 5.125,
    }

    def fetch_period(latitude, longitude, start, end):
        if start == "2015-01-01":
            raise requests.Timeout("connection lost")
        return payload

    monkeypatch.setattr(weather, "_fetch_period", fetch_period)
    monkeypatch.setattr(weather, "RAW_DIR", tmp_path)
    csv_path = tmp_path / "weather_ID003.csv"
    metadata_path = tmp_path / "weather_ID003.metadata.json"
    csv_path.write_text("previous csv", encoding="utf-8")
    metadata_path.write_text("previous metadata", encoding="utf-8")
    args = Namespace(
        system_id=None,
        lat=52.10875,
        lon=5.125278,
        name="ID003",
        start="2014-01-01",
        end="2015-12-31",
    )

    with pytest.raises(RuntimeError, match="2015"):
        weather._cmd(args)

    assert csv_path.read_text(encoding="utf-8") == "previous csv"
    assert metadata_path.read_text(encoding="utf-8") == "previous metadata"
