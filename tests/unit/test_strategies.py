# tests/unit/test_strategies.py

from qmind_quant.core.event_manager import EventManager
from qmind_quant.core.event_types import MarketEvent, SignalEvent
from qmind_quant.strategies.library.ma_crossover_strategy import (
    MovingAverageCrossoverStrategy,
)
from datetime import datetime


def test_ma_crossover_generates_long_signal():
    """
    Tests that a LONG signal is generated when the short MA crosses above the long MA.
    """
    event_manager = EventManager()
    tickers = ["TEST"]
    # Use a very short window for an easy-to-create test case
    strategy = MovingAverageCrossoverStrategy(
        tickers, event_manager, short_window=2, long_window=4
    )

    # 1. Prices are below the long-term average
    event1 = MarketEvent(datetime(2025, 1, 1), "TEST", 100, 100, 100, 100, 1000)
    event2 = MarketEvent(datetime(2025, 1, 2), "TEST", 101, 101, 101, 101, 1000)
    event3 = MarketEvent(datetime(2025, 1, 3), "TEST", 102, 102, 102, 102, 1000)

    # 2. A big price jump that should cause the crossover
    event4 = MarketEvent(datetime(2025, 1, 4), "TEST", 110, 110, 110, 110, 1000)

    # Feed the events to the strategy
    strategy.on_market_event(event1)
    strategy.on_market_event(event2)
    strategy.on_market_event(event3)

    # Before the final event, the queue should be empty
    assert event_manager.empty(), "Queue should be empty before crossover"

    # The final event triggers the signal
    strategy.on_market_event(event4)

    # 3. Assert that a signal was generated
    assert not event_manager.empty(), "A signal should be in the queue"

    signal = event_manager.get()
    assert isinstance(signal, SignalEvent), "Event should be a SignalEvent"
    assert signal.ticker == "TEST", "Signal ticker should be correct"
    assert signal.signal_type == "LONG", "Signal type should be LONG"
