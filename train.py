import logging
import time

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.nodes import FARMReader

if __name__ == "__main__":
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    reader.train(data_dir="annotation",
                 train_filename="train_dataset.json",
                 dev_filename="validation_dataset.json",
                 test_filename="test_dataset.json",
                 use_gpu=False,
                 n_epochs=10,
                 save_dir="models",
                 checkpoint_every=300,
                 checkpoints_to_keep=10,
                 batch_size=8)
