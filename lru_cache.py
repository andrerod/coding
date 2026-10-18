from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self._cache = OrderedDict()
        self._capacity = capacity

    def get(self, key: int) -> int:
        if key not in self._cache:
            # Not found
            return -1

        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)

        self._cache[key] = value


def main():
    lru_cache = LRUCache(2)
    lru_cache.put(1, 1)
    lru_cache.put(2, 2)
    print(lru_cache.get(2))
    print(lru_cache.get(1))
    # Since we accessed 1 last, 2 should be the one getting poped
    lru_cache.put(3, 3)
    print(lru_cache.get(1))
    print(lru_cache.get(2)) # should not exist anymore
    print(lru_cache.get(3))


if __name__ == "__main__":
    main()
