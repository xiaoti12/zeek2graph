import pandas as pd
from typing import List

cipher_list = [5, 47, 61, 4865, 4866, 49171, 49191, 49199, 49200]
cipher_std = {c: i for i, c in enumerate(cipher_list)}


def get_node_attribute(row: pd.Series) -> List:
    attr = []
    attr.append(row["up_bytes"])
    attr.append(row["down_bytes"])
    attr.append(row["up_bytes"] / (row["down_bytes"] + 0.1))
    attr.append(row["up_pkts"] + row["down_pkts"])
    attr.append(row["up_pkts"] / (row["down_pkts"] + 0.1))
    attr.append(row["san_num"])
    attr.append(row["ext_num"])
    attr.append(get_duration(row))
    attr.append(get_tls_version(row))
    attr.append(is_self_sighed(row))
    attr.append(get_valid_time(row))
    attr.append(get_ciphers_len(row))
    attr = attr + get_packet_len_bin(row["or_spl"])
    attr = attr + get_cipher(row)

    return attr


def get_cipher(row: pd.Series) -> List[int]:
    ciphers = [0] * len(cipher_list)
    server_cipher = int(row["cipher"])
    if server_cipher in cipher_std:
        ciphers[cipher_std[server_cipher]] = 1
    return ciphers


def get_packet_len_bin(packet_len: str) -> List:
    packet_len = packet_len.split(",")
    bins = [0 for i in range(10)]
    for l in packet_len:
        index = int(abs(int(l)) / 150)
        bins[min(index, 9)] += 1
    return bins


def get_tls_version(series):
    version_map = {
        (771, 772): 7,
        766: 1,
        767: 2,
        768: 3,
        769: 4,
        770: 5,
        771: 6,
    }
    version = version_map.get((series['server_version'], series['server_supported_version']))
    if version is None:
        version = version_map.get(series['server_version'])
    return version if version is not None else 0


def get_duration(row: pd.Series) -> float:
    # in format of seconds
    delta = row["duration"]
    return delta.total_seconds()


def is_self_sighed(row: pd.Series) -> int:
    if row["subject"] != 0 and row["issuer"] != 0:
        if row["subject"] == row["issuer"]:
            return 1
        else:
            return 0
    else:
        return -1


def get_valid_time(row: pd.Series) -> int:
    if row["valid_time"] == 0:
        return 0
    valid_times = row["valid_time"].split(",")
    valid_times = [float(i) for i in valid_times]
    avg_valid_time = sum(valid_times) / len(valid_times)
    return int(avg_valid_time / 365.0)


def get_ciphers_len(row: pd.Series) -> int:
    ciphers = row["client_ciphers"]
    if ciphers == 0:
        return 0
    ciphers = ciphers.split(",")
    return len(ciphers)
