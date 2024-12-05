from os import path


class LABEL:
    BLACK = 1
    WHITE = 0


class COLUMN:
    SRC_HOST = "id.orig_h"
    DST_HOST = "id.resp_h"
    TIMESTAMP = "ts"
    LABEL = "label"


NODE_INFO_FILE = path.join("raw", "node_info.pkl")

LOG_DIR = "raw_logs"
