from os import path

class LABEL:
    BLACK = 1
    WHITE = 0

class COLUMN:
    HOST = "id.orig_h"
    TIMESTAMP = "ts"
    LABEL = "label"

NODE_INFO_FILE = path.join("raw", "node_info.json")