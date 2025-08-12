# principia_semantica/curvature.py
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from .utils import timeit, logging

@timeit
def compute_ricci_curvature(graph: nx.Graph) -> nx.Graph:
    """
    Computes the Ollivier-Ricci curvature for each edge in the graph.
    
    This operationalizes a key component of Pillar 3, allowing us to analyze
    the local geometry of the conceptual space.

    Args:
        graph: A NetworkX graph.

    Returns:
        The same graph with 'ricciCurvature' attributes added to each edge.
    """
    logging.info("Computing Ollivier-Ricci curvature for the graph.")
    # The alpha parameter controls the distribution of mass from the node to its neighbors.
    # 0.5 is a standard value, representing a "random walk" distribution.
    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    logging.info("Ricci curvature computation complete.")
    return orc.G