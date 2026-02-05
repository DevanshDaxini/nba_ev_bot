# src/utils.py
import time

class SimpleCache:
    def __init__(self, duration=300):
        self.cache = {}
        self.duration = duration # 5 minutes default

    def get(self, key):
        current_time = time.time()
        if key in self.cache:
            timestamp, data = self.cache[key]
            # Check if data is still fresh
            if current_time - timestamp < self.duration:
                return data
        return None

    def set(self, key, data):
        self.cache[key] = (time.time(), data)