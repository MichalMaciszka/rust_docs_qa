import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.utils import SquadData
import pandas as pd
import numpy as np
import math


def read_squads():
    dataset = SquadData.from_file("annotation/dataset/1/answers.json")
    dataset.merge_from_file("annotation/dataset/2/lukaszowe_answersy.json")
    dataset.merge_from_file("annotation/dataset/3/answers.json")
    dataset.merge_from_file("annotation/dataset/4/answers.json")
    dataset.merge_from_file("annotation/dataset/5/answers.json")

    shuffled_dataset = SquadData.to_df(dataset.data).sample(frac=1)

    split_index = math.floor(0.8 * len(shuffled_dataset))
    train = shuffled_dataset[:split_index]
    test = shuffled_dataset[split_index:]

    split_index_validate = math.floor(0.5 * len(test))
    validate = test[:split_index_validate]
    test = test[split_index_validate:]

    train_dataset = SquadData(SquadData.df_to_data(pd.DataFrame(train)))
    test_dataset = SquadData(SquadData.df_to_data(pd.DataFrame(test)))
    validate_dataset = SquadData(SquadData.df_to_data(pd.DataFrame(validate)))

    print(len(dataset.get_all_questions()))
    print(len(train_dataset.get_all_questions()))
    print(len(test_dataset.get_all_questions()))
    print(len(validate_dataset.get_all_questions()))
    

    train_dataset.save("annotation/train_dataset.json")
    test_dataset.save("annotation/test_dataset.json")
    validate_dataset.save("annotation/validation_dataset.json")
    # dataset.save("annotation/final_dataset.json")


if __name__ == "__main__":
    read_squads()