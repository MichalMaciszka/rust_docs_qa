import logging
import time

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.nodes import FARMReader

if __name__ == "__main__":
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    reader.train(data_dir="annotation", train_filename="final_dataset.json", use_gpu=True, n_epochs=3, save_dir=".")
