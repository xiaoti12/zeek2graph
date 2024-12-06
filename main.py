from Combiner import Combiner
from Extractor import Extractor
from Constants import *
import utils
from tqdm import tqdm


if __name__ == "__main__":
    mal2normal = {}
    for malware_log, normal_log in mal2normal.items():
        malware_df = Extractor.log2df(malware_log, LABEL.BLACK, replace_src=False)
        normal_df = Extractor.log2df(normal_log, LABEL.WHITE, replace_src=True)
        combiner = Combiner(malware_df, normal_df)

        combined_df = combiner.get_combined()
        dfs = utils.split_df_by_time(combined_df, "10min")
        for df in tqdm(dfs):
            extractor = Extractor(df=df)
            extractor.extract()
