import networkx as nx

graph = nx.star_graph(20)

nx.write_gexf(graph, "star.gexf")