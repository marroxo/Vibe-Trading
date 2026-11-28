"""Date-only option expiries use the underlying's local trading calendar."""

import pandas as pd
import pytest

from backtest.engines.options_portfolio import OptionPosition, run_options_backtest


@pytest.mark.parametrize("start,expiry", [("2026-03-05", "2026-03-10"), ("2026-10-29", "2026-11-03")])
def test_dated_expiry_matches_naive_calendar_across_dst(tmp_path, start, expiry):
    dates = pd.bdate_range(start, periods=5)
    closes = [100., 101., 102., 103., 104.]

    class Loader:
        name = "local"

        def __init__(self, timezone):
            self.timezone = timezone

        def fetch(self, codes, start_date, end_date):
            return {"SPY": pd.DataFrame(
                {"open": closes, "high": [p + 1 for p in closes],
                 "low": [p - 1 for p in closes], "close": closes, "volume": 10000.},
                index=dates.tz_localize(self.timezone),
            )}

    class Signals:
        def generate(self, data_map):
            return [{
                "date": start, "action": "open", "underlying": "SPY",
                "legs": [{"type": "call", "strike": 100., "expiry": expiry, "qty": 1}],
            }]

    config = {
        "codes": ["SPY"], "source": "local", "engine": "options",
        "start_date": start, "end_date": str(dates[-1].date()),
        "initial_cash": 10000., "commission": 0.,
    }
    for name, zone in [("naive", None), ("aware", "America/New_York")]:
        run_options_backtest(config, Loader(zone), Signals(), tmp_path / name)

    expected = pd.read_csv(tmp_path / "naive/artifacts/trades.csv")
    assert expected["side"].tolist() == ["buy", "exercise"]
    assert expected.iloc[-1]["timestamp"] == expiry
    for filename in ("trades.csv", "equity.csv", "greeks.csv", "metrics.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "naive/artifacts" / filename),
            pd.read_csv(tmp_path / "aware/artifacts" / filename),
        )


def test_explicit_aware_expiry_keeps_its_instant():
    position = OptionPosition(
        "call", 100., "2026-03-10T00:00:00+00:00", 1, 1., "2026-03-05", "SPY",
        calendar_tz="America/New_York",
    )
    same_instant = pd.Timestamp("2026-03-09T20:00:00-04:00")
    assert position.is_expired(same_instant)
    assert position.time_to_expiry(same_instant) == 0


@pytest.mark.parametrize("zone", [None, "America/New_York", "Asia/Hong_Kong"])
def test_date_only_expiry_uses_local_midnight(zone):
    position = OptionPosition("call", 100., "2026-03-10", 1, 1., "2026-03-05", "SPY")
    before = pd.Timestamp("2026-03-09 23:00", tz=zone)
    expiry = pd.Timestamp("2026-03-10", tz=zone)
    assert not position.is_expired(before)
    assert position.is_expired(expiry)
    assert position.time_to_expiry(pd.Timestamp("2026-03-06", tz=zone)) == pytest.approx(4 / 365)


@pytest.mark.parametrize("first_source", ["SPY", "0700.HK"])
def test_partial_position_expiry_does_not_depend_on_source_order(tmp_path, first_source):
    dates = pd.to_datetime(["2026-03-06 00:00", "2026-03-09 12:00", "2026-03-10 12:00"])
    dates = dates.tz_localize("America/New_York")

    class Loader:
        name = "local"

        def fetch(self, codes, start_date, end_date):
            frames = {}
            for code, index, closes in [
                ("SPY", dates, [100., 110., 120.]),
                ("0700.HK", dates.tz_convert("Asia/Hong_Kong"), [50., 50., 50.]),
            ]:
                frames[code] = pd.DataFrame(
                    {"open": closes, "high": [p + 1 for p in closes],
                     "low": [p - 1 for p in closes], "close": closes, "volume": 10000.},
                    index=index,
                )
            other = "0700.HK" if first_source == "SPY" else "SPY"
            return {first_source: frames[first_source], other: frames[other]}

    class Signals:
        def generate(self, data_map):
            leg = {"type": "call", "strike": 100., "expiry": "2026-03-10"}
            return [
                {"date": "2026-03-06", "action": "open", "underlying": "SPY",
                 "legs": [{**leg, "qty": 2}]},
                {"date": "2026-03-06", "action": "close", "underlying": "SPY",
                 "legs": [{**leg, "qty": 1}]},
            ]

    config = {
        "codes": ["SPY", "0700.HK"], "source": "local", "engine": "options",
        "start_date": "2026-03-06", "end_date": "2026-03-10", "initial_cash": 10000.,
        "commission": 0.,
    }
    run_options_backtest(config, Loader(), Signals(), tmp_path)
    trades = pd.read_csv(tmp_path / "artifacts/trades.csv")
    assert trades["side"].tolist() == ["buy", "close", "exercise"]
    # At the second bar Hong Kong is already on March 10, but New York is not.
    # The residual contract must wait for the last bar and settle at 120.
    assert trades.iloc[-1]["price"] == 20.
    assert trades.iloc[-1]["qty"] == 1
