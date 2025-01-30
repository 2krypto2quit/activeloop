from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs.networkx_graph import KG_TRIPLE_DELIMITER
from langchain_core.messages import AIMessage  # Import AIMessage for type checking
from pyvis.network import Network
import networkx as nx
from IPython.core.display import display, HTML
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the text."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property.\n\n"
    "EXAMPLE\n"
    "It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "{text}"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    
    # Extract the content if the response is an AIMessage
    if isinstance(response, AIMessage):
        response_content = response.content
    else:
        response_content = str(response)
    
    return response_content.split(delimiter)

def create_graph_from_triples(triples):
    # Create a directed graph
    G = nx.DiGraph()
    
    for triple in triples:
        subject, predicate, obj = triple.strip().split(',')
        G.add_edge(subject.strip(), obj.strip(), label=predicate.strip())
    
    return G

chain = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT | llm

text = "The city of Paris is the capital and most populous city of France. The Eiffel Tower is a famous landmark in Paris."
triples = chain.invoke({"text": text})  # Pass input as a dictionary
triples_list = parse_triples(triples)
print(triples_list)

 #Convert the NetworkX graph to a PyVis network
def nx_to_pyvis(networkx_graph):
    pyvis_graph = Network(notebook=True, cdn_resources='remote')
    for node in networkx_graph.nodes():
        pyvis_graph.add_node(node)
    for edge in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(edge[0], edge[1], label=edge[2]["label"])
    return pyvis_graph

triplets = [t.strip() for t in triples_list if t.strip()]
graph = create_graph_from_triples(triplets)
pyvis_network = nx_to_pyvis(graph)

# Customize the appearance of the graph
pyvis_network.toggle_hide_edges_on_drag(True)
pyvis_network.toggle_physics(False)
pyvis_network.set_edge_smooth('discrete')

# Show the interactive knowledge graph visualization
pyvis_network.show('knowledge_graph.html')
display(HTML('knowledge_graph.html'))