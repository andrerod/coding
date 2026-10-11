import struct
import threading


class PersistentKVStore:
    def __init__(self):
        self._store = {}
        self._locks = {}
        self._meta_lock = threading.Lock()  # Protects the _locks dictionary creation

    def get_key_lock(self, key:str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()

            return self._locks[key]

    def set(self, key: str, value: str) -> None:
        key_lock = self.get_key_lock(key)
        with key_lock:
            self._store[key] = value

    def get(self, key: str) -> str:
        key_lock = self.get_key_lock(key)
        with key_lock:
            return self._store.get(key, None)

    def delete(self, key: str) -> bool:
        key_lock = self.get_key_lock(key)
        with key_lock:
            with self._meta_lock:
                if key in self._store:
                    del self._store[key]
                    self._locks.pop(key, None)
                    return True

            return False

    def serialize(self) -> bytes:
        with self._meta_lock:
            # Format [4 char version] + per entry: [Key Length (4B)] + [Key (UTF-8)] + [Val Length (4B)] + [Val (UTF-8)]
            blob = bytearray()

            version_bytes = "KVS1".encode("utf-8")
            blob.extend(version_bytes)

            for key, value in self._store.items():
                key_bytes = key.encode("utf-8")
                value_bytes = value.encode("utf-8")

                blob.extend(struct.pack("!I", len(key_bytes)))
                blob.extend(key_bytes)
                blob.extend(struct.pack("!I", len(value_bytes)))
                blob.extend(value_bytes)

            return bytes(blob)


    def deserialize(self, data: bytes) -> None:
        with self._meta_lock:
            version = data[0:4].decode("utf-8")
            if version != "KVS1":
                raise Exception(f"Invalid version {version}")

            curr_pos = 4
            while curr_pos < len(data):
                if curr_pos + 4 >= len(data):
                    raise Exception("Malformed key length")
                key_length = struct.unpack("!I", data[curr_pos:curr_pos + 4])[0]
                curr_pos += 4

                key = data[curr_pos:curr_pos + int(key_length)].decode("utf-8")
                curr_pos += key_length

                if curr_pos + 4 >= len(data):
                    raise Exception("Malformed value length")
                value_length = struct.unpack("!I", data[curr_pos:curr_pos + 4])[0]
                curr_pos += 4

                value = data[curr_pos:curr_pos + int(value_length)].decode("utf-8")
                curr_pos += value_length

                self._store[key] = value

        return

def main():
    kv_store = PersistentKVStore()
    kv_store.set("foo", "1")
    print(f"foo: {kv_store.get("foo")}")
    kv_ser = kv_store.serialize()
    kv_store2 = PersistentKVStore()
    kv_store2.deserialize(kv_ser)
    print(f"foo: {kv_store2.get("foo")}")
    print(kv_store2.delete("foo"))
    print(kv_store2.delete("foo"))


if __name__ == "__main__":
    main()
