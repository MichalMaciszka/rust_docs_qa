import logging
import time

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger("haystack")

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import MarkdownConverter, TextConverter
from haystack.pipelines import Pipeline, ExtractiveQAPipeline
from pathlib import Path
from haystack.nodes import FARMReader, TransformersReader, EmbeddingRetriever, BM25Retriever, DensePassageRetriever
from haystack.nodes.file_classifier import FileTypeClassifier
from haystack.nodes.preprocessor import PreProcessor
from haystack.utils import print_answers
from haystack.utils import launch_es
from haystack.schema import EvaluationResult, MultiLabel




# langchain
# sbert
"""
I indexing pipeline:
    1. file type classifier
    2. text converter
    3. preprocessor
    4. document store
II. query pipeline:
    1. retriever
    2. reader
    3. prompt
"""
def basic_qa():
    paths = [p for p in Path("rust_txt").glob("**/*")]

    # document_store = InMemoryDocumentStore(use_bm25=True, use_gpu=True, similarity='dot_product')
    document_store = InMemoryDocumentStore(use_bm25=False,
        use_gpu=True,
        similarity='dot_product',
        return_embedding=True,
        embedding_dim=768
    )
    indexing_pipeline = Pipeline()

    classifier = FileTypeClassifier(supported_types=["txt"])
    indexing_pipeline.add_node(classifier, name="Classifier", inputs=["File"])

    converter = TextConverter()
    indexing_pipeline.add_node(converter, name="Converter", inputs=["Classifier.output_1"])

    preprocessor = PreProcessor(
        clean_whitespace=False,
        clean_empty_lines=False,
        split_length=768,
        split_overlap=0,
        split_respect_sentence_boundary=False,
    )
    indexing_pipeline.add_node(preprocessor, name="Preprocessor", inputs=["Converter"])

    indexing_pipeline.add_node(document_store, name="Document store", inputs=["Preprocessor"])

    indexing_pipeline.run(file_paths=paths)

    # retriever = BM25Retriever(document_store=document_store)
    # retriever = DensePassageRetriever.load(load_dir="./dpr_models", document_store=document_store)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        # embedding_model="sentence-transformers/multi-qa-mpnet-base-cos-v1"
        # embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
        embedding_model="./embedding_model"
    )
    # retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="facebook/dpr-question_encoder-single-nq-base", passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")
    # reader = FARMReader(model_name_or_path="./model_checkpoints/epoch_10_step_0", use_gpu=True)
    reader = FARMReader(model_name_or_path="reader_models", use_gpu=True)
    # retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2", use_gpu=True)
    # reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    # reader = TransformersReader(model_name_or_path="impira/layoutlm-document-qa", use_gpu=False)
    # prompt_model = PromptModel()
    # prompt_node = PromptNode(prompt_model, default_prompt_template="deepset/question-answering-per-document")

    # document_store.delete_documents()
    # document_store.write_documents(doc['documents'])
    document_store.update_embeddings(retriever)

    document_store.add_eval_data(
        filename="annotation/test_dataset.json",
        preprocessor=preprocessor
    )

    # Query Pipeline
    # document_store.update_embeddings(retriever)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])
    # pipeline.add_node(component=prompt_node, name="Prompt node", inputs=["Reader"])


    eval_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)
    print("-----------------------------------------------")
    print(eval_labels)
    print(document_store.get_label_count())


    eval_result = pipeline.eval(labels=eval_labels, add_isolated_node_eval=False, params={"Retriever": {"top_k": 5}})
   
    reader_result = eval_result["Reader"]
    reader_result.head()

    retriever_result = eval_result["Retriever"]
    retriever_result.head()

    query = "What is TCP protocol?"
    retriever_book_of_life = retriever_result[retriever_result["query"] == query]
    reader_book_of_life = reader_result[reader_result["query"] == query]
    eval_result.save("../")

    saved_eval_result = EvaluationResult.load("../")
    metrics = saved_eval_result.calculate_metrics()
    print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
    print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
    print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
    print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
    print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

    print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
    print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')
    
    pipeline.print_eval_report(saved_eval_result)

    eval_pipe = Pipeline()
    eval_store = InMemoryDocumentStore(index="scifact_beir", use_bm25=True, use_gpu=True)
    eval_pipe.add_node(classifier, name="Classifier", inputs=["File"])
    eval_pipe.add_node(converter, name="Converter", inputs=["Classifier.output_1"])
    eval_pipe.add_node(preprocessor, name="Preprocessor", inputs=["Converter"])
    eval_pipe.add_node(eval_store, name="DS", inputs=["Preprocessor"])
    
    # retriever.document_store = eval_store
    # print(20 * '-' + "\n beir:")
    # ndcg, _map, recall, precision = Pipeline.eval_beir(
    #     index_pipeline=eval_pipe, query_pipeline=pipeline, dataset="scifact"
    # )
    # print(ndcg, _map, recall, precision)
    # print(20 * '-')

    retriever.document_store = document_store



    while True:
        question = input("Question: ")
        prediction = pipeline.run(
            query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}
        )
        print_answers(prediction, details="medium")

if __name__ == "__main__":
   # launch_es()
    basic_qa()
