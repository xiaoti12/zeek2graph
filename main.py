from Combiner import Combiner
from Extractor import Extractor
from Constants import *


if __name__ == "__main__":
    malware_df = Extractor.log2df("mta-2023.log", LABEL.BLACK)
    normal_df = Extractor.log2df("campus1.log", LABEL.WHITE)
    combiner = Combiner(malware_df, normal_df)

    combined_df = combiner.get_combined()
    extractor = Extractor(df=combined_df)

    extractor.extract()
