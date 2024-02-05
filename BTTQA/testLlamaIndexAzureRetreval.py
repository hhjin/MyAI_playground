from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SentenceSplitter
import os
import nest_asyncio
nest_asyncio.apply()

OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE=os.environ["OPENAI_API_BASE"]
OPENAI_API_VERSION=os.environ["OPENAI_API_VERSION"]


documents = SimpleDirectoryReader("rawtext/").load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

# by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
# 使用Azure OpenAI Service
azure_lm = AzureOpenAI(engine="gpt35turbo-16k",  temperature=0.0, azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)
azure_embeddings=AzureOpenAIEmbedding(model="embedding2", azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)

service_context = ServiceContext.from_defaults(llm=azure_lm, embed_model=azure_embeddings )

vector_index = VectorStoreIndex(nodes[:1], service_context=service_context )

retriever = vector_index.as_retriever(similarity_top_k=2)

from llama_index.response.notebook_utils import display_source_node
retrieved_nodes = retriever.retrieve("What did the author do growing up?")
#retrieved_nodes = retriever.retrieve("How did Y Combinator challenge the traditional notion of \"deal flow\" and contribute to the creation of new startups that would not have existed otherwise?")
for node in retrieved_nodes:
    display_source_node(node, source_length=1000)
    print(node)

