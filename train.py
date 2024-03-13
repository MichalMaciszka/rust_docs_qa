import logging
import torch

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.nodes import FARMReader

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())


    print("###################################################################################")


    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    reader.train(data_dir="annotation",
                 train_filename="train_dataset.json",
                 dev_filename="validation_dataset.json",
                 test_filename="test_dataset.json",
                 use_gpu=True,
                 n_epochs=100,
                 checkpoint_every=100000,
                 checkpoints_to_keep=100,
                 save_dir="reader_models",
                 evaluate_every=2223,
                 batch_size=10)
