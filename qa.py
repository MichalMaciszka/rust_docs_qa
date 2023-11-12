import logging
import time

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import MarkdownConverter
from haystack.pipelines import Pipeline, ExtractiveQAPipeline
from pathlib import Path
from haystack.nodes import FARMReader, TransformersReader, EmbeddingRetriever, BM25Retriever
from haystack.nodes.file_classifier import FileTypeClassifier
from haystack.nodes.preprocessor import PreProcessor
from haystack.utils import print_answers




# langchain
# sbert
"""
I indexing pipeline:
    1. file type classifier
    2. markdown converter
    3. preprocessor
    4. document store
II. query pipeline:
    1. retriever
    2. reader
    3. prompt
"""
def basic_qa():
    paths = [p for p in Path("rust_book").glob("**/*")]

    document_store = InMemoryDocumentStore(use_bm25=True, use_gpu=True)
    indexing_pipeline = Pipeline()

    classifier = FileTypeClassifier(supported_types=["md"])
    indexing_pipeline.add_node(classifier, name="Classifier", inputs=["File"])

    converter = MarkdownConverter()
    indexing_pipeline.add_node(converter, name="Converter", inputs=["Classifier.output_1"])

    preprocessor = PreProcessor(
        clean_whitespace=False,
        clean_empty_lines=False,
        split_length=30,
        split_overlap=10,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline.add_node(preprocessor, name="Preprocessor", inputs=["Converter"])

    indexing_pipeline.add_node(document_store, name="Document store", inputs=["Preprocessor"])

    indexing_pipeline.run(file_paths=paths)

    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="annotation/final_model", use_gpu=True)
    # retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2", use_gpu=True)
    # reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    # reader = TransformersReader(model_name_or_path="impira/layoutlm-document-qa", use_gpu=False)
    # prompt_model = PromptModel()
    # prompt_node = PromptNode(prompt_model, default_prompt_template="deepset/question-answering-per-document")

    # document_store.update_embeddings(retriever)

    # Query Pipeline
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])
    # pipeline.add_node(component=prompt_node, name="Prompt node", inputs=["Reader"])

    while True:
        question = input("Question: ")
        prediction = pipeline.run(
            query=question, params={"Retriever": {"top_k": 100}, "Reader": {"top_k": 7}}
        )
        print_answers(prediction, details="minimum")

def different_qa():
    reader = TransformersReader("ahotrod/albert_xxlargev1_squad2_512", use_gpu=True)
    paths = [p for p in Path("rust_book").glob("**/*")]
    logger.warning(paths)
    time.sleep(10)

    document_store = InMemoryDocumentStore(use_bm25=True, use_gpu=True)
    indexing_pipeline = Pipeline()

    classifier = FileTypeClassifier(supported_types=["md"])
    indexing_pipeline.add_node(classifier, name="Classifier", inputs=["File"])
    converter = MarkdownConverter()
    indexing_pipeline.add_node(converter, name="Converter", inputs=["Classifier.output_1"])

    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_empty_lines=True,
        split_length=100,
        split_overlap=50,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline.add_node(preprocessor, name="Preprocessor", inputs=["Converter"])

    indexing_pipeline.add_node(document_store, name="Document store", inputs=["Preprocessor"])

    indexing_pipeline.run(file_paths=paths)
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2")
    pipeline = ExtractiveQAPipeline(reader, retriever)
    document_store.update_embeddings(retriever=retriever)

    while True:
        question = input("Question: ")
        prediction = pipeline.run(
            query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )
        print_answers(prediction, details="all")
        

if __name__ == "__main__":
    # launch_es()
    basic_qa()
    # different_qa()
