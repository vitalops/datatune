from typing import Dict, List, Any, Tuple
import zlib
import json


def compress_key_map_keys(key_map: Dict[str, str]) -> bytes:
    ret = b"".join(length.to_bytes(4, "big") for length in map(len, key_map))
    return ret + zlib.compress(b"".join(k.encode("utf-8") for k in key_map))

def decompress_key_map_keys(data: bytes, num_keys: int) -> List[str]:
    mview = memoryview(data)
    lengths = [int.from_bytes(mview[i:i+4], "big") for i in range(0, 4 * num_keys, 4)]
    mview = memoryview(zlib.decompress(mview[4 * num_keys:]))
    ret = []
    idx = 0
    for length in lengths:
        ret.append(str(mview[idx:idx+length], "utf-8"))
        idx += length
    return ret

def compress_key_map_values(key_map: Dict[str, str]) -> bytes:
    transactions: Dict[str, int] = {}
    idxs: List[int] = []
    for val in key_map.values():
        if val not in transactions:
            transactions[val] = len(transactions)
        idxs.append(transactions[val])
    ret = len(transactions).to_bytes(4, "big")
    for transaction in transactions:
        byts = transaction.encode("utf-8")
        ret += len(byts).to_bytes(4, "big")
        ret += byts
    ret += b"".join(idx.to_bytes(4, "big") for idx in idxs)
    return ret

def decompress_key_map_values(data: bytes, num_keys: int) -> List[str]:
    mview = memoryview(data)
    num_transactions = int.from_bytes(mview[:4], "big")
    idx = 4
    transactions = []
    for _ in range(num_transactions):
        transaction_len = int.from_bytes(mview[idx:idx+4], "big")
        idx += 4
        transactions.append(str(mview[idx:idx+transaction_len], "utf-8"))
        idx += transaction_len
    idxs = [int.from_bytes(mview[i:i+4], "big") for i in range(idx, idx + 4 * num_keys, 4)]
    return [transactions[idx] for idx in idxs]

def key_map_to_bytes(key_map: Dict[str, str]) -> bytes:
    ret = len(key_map).to_bytes(4, "big")
    keys_compressed = compress_key_map_keys(key_map)
    values_compressed = compress_key_map_values(key_map)
    ret += len(keys_compressed).to_bytes(4, "big")
    ret += len(values_compressed).to_bytes(4, "big")
    ret += keys_compressed
    ret += values_compressed
    return ret

def key_map_from_bytes(data: bytes) -> dict:
    mview = memoryview(data)
    num_keys = int.from_bytes(mview[:4], "big")
    idx = 4
    keys_len = int.from_bytes(mview[idx:idx+4], "big")
    idx += 4
    values_len = int.from_bytes(mview[idx:idx+4], "big")
    idx += 4
    keys = decompress_key_map_keys(mview[idx:idx+keys_len], num_keys)
    idx += keys_len
    values = decompress_key_map_values(mview[idx:idx+values_len], num_keys)
    return dict(zip(keys, values))


def commit_metadata_to_bytes(metadata: Dict[str, Any]) -> bytes:
    return json.dumps(metadata).encode("utf-8")


def commit_metadata_from_bytes(data: bytes) -> Dict[str, Any]:
    return json.loads(data)


def commit_to_bytes(key_map: Dict[str, str], metadata: Dict[str, Any]) -> bytes:
    key_map_bytes = key_map_to_bytes(key_map)
    metadata_bytes = commit_metadata_to_bytes(metadata)
    return len(key_map_bytes).to_bytes(4, "big") + key_map_bytes + metadata_bytes


def commit_from_bytes(data: bytes) -> Tuple[Dict[str, str], Dict[str, Any]]:
    mview = memoryview(data)
    key_map_len = int.from_bytes(mview[:4], "big")
    key_map = key_map_from_bytes(mview[4:4+key_map_len])
    metadata = commit_metadata_from_bytes(bytes(mview[4+key_map_len:]))
    return key_map, metadata
