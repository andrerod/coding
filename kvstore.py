import struct


class PersistentKVStore:
    def __init__(self):
        self.store = {}

    def set(self, key: str, value: str) -> None:
        """Stores or updates a string value for a string key."""
        self.store[key] = value

    def get(self, key: str) -> str:
        """Retrieves the value, or returns None if it doesn't exist."""
        return self.store.get(key, None)

    def serialize(self) -> bytes:
        """
        Serializes the KV store into a safe binary block using length-prefixing.
        Format per entry: [Key Length (4B)] + [Key (UTF-8)] + [Val Length (4B)] + [Val (UTF-8)]
        """
        blob = bytearray()
        for key, value in self.store.items():
            key_bytes = key.encode("utf-8")
            val_bytes = value.encode("utf-8")

            # Pack lengths as 4-byte big-endian integers (!I)
            blob.extend(struct.pack("!I", len(key_bytes)))
            blob.extend(key_bytes)
            blob.extend(struct.pack("!I", len(val_bytes)))
            blob.extend(val_bytes)

        return bytes(blob)

    def deserialize(self, blob: bytes) -> None:
        """Rebuilds the entire store state from the given byte stream."""
        new_store = {}
        offset = 0
        blob_len = len(blob)

        try:
            while offset < blob_len:
                # 1. Read Key Length
                if offset + 4 > blob_len:
                    raise ValueError("Malformed blob: Missing key length header.")
                key_len = struct.unpack("!I", blob[offset : offset + 4])[0]
                offset += 4

                # 2. Read Key Bytes
                if offset + key_len > blob_len:
                    raise ValueError("Malformed blob: Truncated key payload.")
                key = blob[offset : offset + key_len].decode("utf-8")
                offset += key_len

                # 3. Read Value Length
                if offset + 4 > blob_len:
                    raise ValueError("Malformed blob: Missing value length header.")
                val_len = struct.unpack("!I", blob[offset : offset + 4])[0]
                offset += 4

                # 4. Read Value Bytes
                if offset + val_len > blob_len:
                    raise ValueError("Malformed blob: Truncated value payload.")
                value = blob[offset : offset + val_len].decode("utf-8")
                offset += val_len

                new_store[key] = value

            self.store = new_store
        except Exception as e:
            raise ValueError(f"Deserialization failed due to corrupt input: {e}")


import bisect
from typing import Dict, List, Optional, Tuple


class TimeBasedKVStore:
    def __init__(self):
        # Maps Key -> List of Tuple(Timestamp, Value)
        self.store: Dict[str, List[Tuple[int, str]]] = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.store:
            self.store[key] = []

        # If timestamps arrive strictly in-order, a simple append works:
        # self.store[key].append((timestamp, value))

        # If timestamps arrive OUT-OF-ORDER (Common OpenAI follow-up), maintain sorting:
        entries = self.store[key]
        idx = bisect.bisect_left(entries, (timestamp, ""))

        if idx < len(entries) and entries[idx][0] == timestamp:
            # Overwrite if exact same timestamp exists
            entries[idx] = (timestamp, value)
        else:
            entries.insert(idx, (timestamp, value))

    def get(self, key: str, timestamp: int) -> Optional[str]:
        if key not in self.store:
            return None

        entries = self.store[key]

        # Binary search for the largest timestamp <= requested timestamp
        # bisect_right finds the insertion point after any entries matching the timestamp
        idx = bisect.bisect_right(
            entries, (timestamp, chr(1114111))
        )  # High character ceiling for comparison

        if idx == 0:
            return None  # No version exists before or at this timestamp

        return entries[idx - 1][1]


from typing import Dict, List, Optional


class TransactionalKVStore:
    def __init__(self):
        # The ultimate source of truth
        self.global_store: Dict[str, str] = {}
        # A stack of transaction state logs.
        # Each active transaction tracks its local additions/deletions.
        self.transaction_stack: List[Dict[str, Optional[str]]] = []

    def get(self, key: str) -> Optional[str]:
        # 1. Search from the most recent active nested transaction backwards
        for tx in reversed(self.transaction_stack):
            if key in tx:
                val = tx[key]
                return (
                    None if val is None else val
                )  # None represents a deleted key (tombstone)

        # 2. If not found in any transaction, fallback to the global store
        return self.global_store.get(key, None)

    def set(self, key: str, value: str) -> None:
        if self.transaction_stack:
            # Stage the change locally in the active transaction block
            self.transaction_stack[-1][key] = value
        else:
            # No active transaction; modify the global state immediately
            self.global_store[key] = value

    def delete(self, key: str) -> None:
        if self.transaction_stack:
            # Use None as a tombstone marker to signify deletion inside this transaction
            self.transaction_stack[-1][key] = None
        else:
            if key in self.global_store:
                del self.global_store[key]

    def begin(self) -> None:
        # Push a brand-new local transaction frame onto the stack
        self.transaction_stack.append({})

    def commit(self) -> None:
        if not self.transaction_stack:
            raise RuntimeError("No active transaction to commit.")

        # Pop the current transaction frame
        completed_tx = self.transaction_stack.pop()

        if self.transaction_stack:
            # Merge changes up into the parent nested transaction frame
            self.transaction_stack[-1].update(completed_tx)
        else:
            # We are at the root level; flush changes permanently into the global store
            for key, val in completed_tx.items():
                if val is None:
                    self.global_store.pop(key, None)
                else:
                    self.global_store[key] = val

    def rollback(self) -> None:
        if not self.transaction_stack:
            raise RuntimeError("No active transaction to rollback.")
        # Discard the latest transaction state entirely
        self.transaction_stack.pop()


import os
import struct
from typing import Dict, Optional, Tuple


class ShardedDiskKVStore:
    def __init__(self, data_dir: str, max_shard_size_bytes: int = 1024):
        self.data_dir = data_dir
        self.max_shard_size = max_shard_size_bytes

        # In-memory index: key -> (shard_id, byte_offset)
        self.index: Dict[bytes, Tuple[int, int]] = {}

        # Storage tracking
        self.current_shard_id = 0
        self.current_shard_file = None
        self.current_shard_size = 0

        # Initialize storage directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Recover database state from prior crashes/restarts
        self._recover_and_initialize()

    def _get_shard_path(self, shard_id: int) -> str:
        return os.path.join(self.data_dir, f"shard_{shard_id}.db")

    def _recover_and_initialize(self) -> None:
        """Finds all existing shards on disk and rebuilds the index linearly."""
        shard_ids = []
        for filename in os.listdir(self.data_dir):
            if filename.startswith("shard_") and filename.endswith(".db"):
                try:
                    shard_id = int(filename.split("_")[1].split(".")[0])
                    shard_ids.append(shard_id)
                except ValueError:
                    continue

        if not shard_ids:
            # Clean slate: Initialize the first shard
            self._rotate_shard(0)
            return

        # Scan shards in sequential order to process overwrites correctly
        shard_ids.sort()
        for s_id in shard_ids:
            path = self._get_shard_path(s_id)
            file_size = os.path.getsize(path)

            with open(path, "rb") as f:
                offset = 0
                while offset < file_size:
                    # Read 8-byte header: key_len (4B), val_len (4B)
                    header = f.read(8)
                    if not header or len(header) < 8:
                        break  # End of file or truncated record

                    k_len, v_len = struct.unpack(">II", header)

                    # Record the exact starting offset of this valid payload
                    self.index[f.read(k_len)] = (s_id, offset)

                    # Skip past the value bytes to move to the next entry
                    f.seek(v_len, os.SEEK_CUR)
                    offset += 8 + k_len + v_len

        # Resume writing at the tail of the latest active shard
        self._rotate_shard(shard_ids[-1])

    def _rotate_shard(self, shard_id: int) -> None:
        """Closes the current shard file handle and opens the designated shard."""
        if self.current_shard_file:
            self.current_shard_file.close()

        self.current_shard_id = shard_id
        path = self._get_shard_path(shard_id)

        # Open in append + binary mode ('a+b' or 'r+b' if it already exists)
        if os.path.exists(path):
            self.current_shard_file = open(path, "r+b")
            self.current_shard_file.seek(0, os.SEEK_END)
            self.current_shard_size = self.current_shard_file.tell()
        else:
            self.current_shard_file = open(path, "w+b")
            self.current_shard_size = 0

    def put(self, key: bytes, value: bytes) -> None:
        if not isinstance(key, bytes) or not isinstance(value, bytes):
            raise TypeError("Keys and values must be bytes.")

        # Format: [Key Len (4B)][Val Len (4B)][Key Data][Value Data]
        entry_size = 8 + len(key) + len(value)

        if entry_size > self.max_shard_size:
            raise ValueError("Payload size exceeds maximum individual shard capacity.")

        # Shard Rotation Rule: If entry doesn't fit in current shard, open a new one
        if self.current_shard_size + entry_size > self.max_shard_size:
            self._rotate_shard(self.current_shard_id + 1)

        # Capture current file offset before writing the data
        offset = self.current_shard_file.tell()

        # Write to disk
        header = struct.pack(">II", len(key), len(value))
        self.current_shard_file.write(header)
        self.current_shard_file.write(key)
        self.current_shard_file.write(value)
        self.current_shard_file.flush()  # Commit buffer to disk

        # Update trackers
        self.index[key] = (self.current_shard_id, offset)
        self.current_shard_size += entry_size

    def get(self, key: bytes) -> Optional[bytes]:
        if key not in self.index:
            return None

        shard_id, offset = self.index[key]

        # Random access read from the specific shard file
        path = self._get_shard_path(shard_id)
        with open(path, "rb") as f:
            f.seek(offset)
            header = f.read(8)
            k_len, v_len = struct.unpack(">II", header)

            # Skip past the key bytes directly to the value
            f.seek(k_len, os.SEEK_CUR)
            return f.read(v_len)

    def close(self) -> None:
        """Safely flushes and closes active descriptors."""
        if self.current_shard_file:
            self.current_shard_file.close()


def main():
    kv_store = PersistentKVStore()
    kv_store.set("foo", "1")
    print(f"foo: {kv_store.get("foo")}")
    kv_ser = kv_store.serialize()
    kv_store2 = PersistentKVStore()
    kv_store2.deserialize(kv_ser)
    print(f"foo: {kv_store2.get("foo")}")


if __name__ == "__main__":
    main()
