import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.utils import SquadData


def read_squads():
    dataset = SquadData.from_file("annotation/dataset/1/answers.json")
    dataset.merge_from_file("annotation/dataset/2/lukaszowe_answersy.json")
    dataset.merge_from_file("annotation/dataset/3/answers.json")
    dataset.merge_from_file("annotation/dataset/4/answers.json")
    dataset.merge_from_file("annotation/dataset/5/answers.json")
    print(len(dataset.get_all_questions()))
    dataset.save("annotation/final_dataset.json")


if __name__ == "__main__":
    read_squads()