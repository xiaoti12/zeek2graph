from Combiner import Combiner
from Extractor import Extractor
from Constants import *
import utils
from tqdm import tqdm


if __name__ == "__main__":
    malware_df = Extractor.log2df("mta-2023.log", LABEL.BLACK, replace_src=False)
    normal_df = Extractor.log2df("campus1.log", LABEL.WHITE, replace_src=True)
    combiner = Combiner(malware_df, normal_df)

    combined_df = combiner.get_combined()
    dfs = utils.split_df_by_time(combined_df, "30min")
    for df in tqdm(dfs):
        extractor = Extractor(df=df)
        extractor.extract()
