class ResumableIterator:
    def __init__(self, data):
        self.data = list(data)
        self.index = 0
        self.is_paused = False
        self.is_exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_paused:
            raise StopIteration("Iterator is currently paused. Call .resume() first.")

        if self.index >= len(self.data):
            self.is_exhausted = True
            raise StopIteration("End of iterator reached.")

        item = self.data[self.index]
        self.index += 1
        return item

    def pause(self):
        """Pause iteration."""
        self.is_paused = True

    def resume(self):
        """Resume iteration if paused."""
        self.is_paused = False

    def skip(self, n: int = 1):
        """Skip the next n items."""
        if self.is_paused:
            raise RuntimeError("Cannot skip while paused. Resume the iterator first.")
        self.index = min(self.index + n, len(self.data))
        if self.index >= len(self.data):
            self.is_exhausted = True

    def reset(self):
        """Reset the iterator back to the beginning."""
        self.index = 0
        self.is_paused = False
        self.is_exhausted = False

    def get_state(self) -> dict:
        """Export the current state as a serializable dictionary."""
        return {
            "index": self.index,
            "is_paused": self.is_paused,
            "is_exhausted": self.is_exhausted,
        }

    def set_state(self, state: dict):
        """Restore the iterator position and state from a checkpoint."""
        self.index = state.get("index", 0)
        self.is_paused = state.get("is_paused", False)
        self.is_exhausted = state.get("is_exhausted", False)

def main():
    data = [1, 2, 3, 4, 5]
    itr = ResumableIterator(data)
    print(next(itr))
    print(next(itr))
    itr.pause()
    try:
        print(next(itr))
    except:
        print("Correctly rejected next while paused")
    itr.resume()
    print(next(itr))


if __name__ == "__main__":
    main()
