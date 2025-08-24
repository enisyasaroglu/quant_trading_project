# qmind_quant/core/event_manager.py

from queue import Queue
from qmind_quant.core.event_types import Event


class EventManager:
    """
    Coordinates all events between the components of the system.
    """

    def __init__(self):
        self.events = Queue()

    def put(self, event: Event):
        """
        Puts an event in the queue.
        """
        self.events.put(event)

    def get(self) -> Event:
        """
        Gets an event from the queue.
        """
        return self.events.get()

    def empty(self) -> bool:
        """
        Checks if the event queue is empty.
        """
        return self.events.empty()
