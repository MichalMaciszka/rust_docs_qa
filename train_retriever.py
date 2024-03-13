
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
query_model = "bert-base-uncased"
passage_model = "bert-base-uncased"
doc_dir = "./dpr_data"
train_filename = "train_dpr_out.json"
dev_filename = "validation_dpr_out.json"
test_filename = "test_dpr_out.json"
save_dir = "./dpr_models/"

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


retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    # max_seq_len_query=64,
    max_seq_len_passage=512,
    batch_size=16,
    use_gpu=True,
    similarity_function = "dot_product",
    # use_fast_tokenizers=True
)

retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    test_filename=test_filename,
    dev_filename=dev_filename,
    n_epochs=100,
    batch_size=1,
    grad_acc_steps=8,
    save_dir=save_dir,
    evaluate_every=3000,
    embed_title=True,
    checkpoints_to_keep=30,
    use_amp=True
)
