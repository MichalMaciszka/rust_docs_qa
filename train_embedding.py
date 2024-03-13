import json

def squad_to_embedding_train(squad_path: str) -> list:
    result = []

    with open(squad_path) as f:
        d = json.load(f)
        # print(d['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['text'])
        data = d['data']
        for document in data:
            paragraphs = document['paragraphs']
            for par in paragraphs:
                ctx = par['context']
                qas = par['qas']
                for qa in qas:
                    q = qa['question']
                    result.append({'question': q, 'pos_doc': ctx})
    print(f"dupa:{len(result)}")
    return result

import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import MarkdownConverter, TextConverter
from haystack.pipelines import Pipeline, ExtractiveQAPipeline
from pathlib import Path
from haystack.nodes import FARMReader, TransformersReader, EmbeddingRetriever, BM25Retriever
from haystack.nodes.file_classifier import FileTypeClassifier
from haystack.nodes.preprocessor import PreProcessor
from haystack.utils import print_answers
from haystack.utils import launch_es
from haystack.schema import EvaluationResult, MultiLabel

# query_model = "facebook/dpr-question_encoder-single-nq-base"
# passage_model = "facebook/dpr-ctx_encoder-single-nq-base"
# query_model = "bert-base-uncased"
# passage_model = "bert-base-uncased"
# doc_dir = "./dpr_data"
# train_filename = "train_dpr_out.json"
# dev_filename = "validation_dpr_out.json"
# test_filename = "test_dpr_out.json"
# save_dir = "./dpr_models/"

# ------------------------------------------------------------------

paths = [p for p in Path("rust_txt").glob("**/*")]

document_store = InMemoryDocumentStore(use_bm25=False, use_gpu=True, similarity='dot_product')
indexing_pipeline = Pipeline()

classifier = FileTypeClassifier(supported_types=["txt"])
indexing_pipeline.add_node(classifier, name="Classifier", inputs=["File"])

converter = TextConverter()
indexing_pipeline.add_node(converter, name="Converter", inputs=["Classifier.output_1"])

preprocessor = PreProcessor(
    clean_whitespace=False,
    clean_empty_lines=False,
    split_length=500,
    split_overlap=0,
    split_respect_sentence_boundary=False,
)
indexing_pipeline.add_node(preprocessor, name="Preprocessor", inputs=["Converter"])

indexing_pipeline.add_node(document_store, name="Document store", inputs=["Preprocessor"])
# indexing_pipeline.add_node(document_store, name="Document store", inputs=["Converter"])

indexing_pipeline.run(file_paths=paths)

retriever = EmbeddingRetriever(
    document_store=document_store,
    # embedding_model="sentence-transformers/multi-qa-mpnet-base-cos-v1"
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)

# document_store.update_embeddings(retriever)

retriever.train(
    training_data=squad_to_embedding_train("annotation/train_dataset.json"),
    learning_rate=1e-5,
    n_epochs=100,
    batch_size=8,
    save_best_model=True,
    checkpoint_save_steps=1000
)

retriever.save("embedding_model")
