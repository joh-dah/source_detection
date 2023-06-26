import networkx as nx
from scipy.io import mmread
import os
import torch_geometric

def main():
    data_dir = "./real_data"
    os.makedirs(data_dir, exist_ok=True)

    airports = torch_geometric.datasets.Airports(root=data_dir, name="USA")                     #nodes: 1190,  edges: 13599,  avg(degree): 22.86, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Airports.html#torch_geometric.datasets.Airports
    wiki = torch_geometric.datasets.AttributedGraphDataset(root=data_dir, name="Wiki")          #nodes: 2405,  edges: 17981,  avg(degree): 13.74, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.AttributedGraphDataset.html#torch_geometric.datasets.AttributedGraphDataset
    facebook = torch_geometric.datasets.AttributedGraphDataset(root=data_dir, name="Facebook")  #nodes: 4039,  edges: 88234,  avg(degree): 43.69, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.AttributedGraphDataset.html#torch_geometric.datasets.AttributedGraphDataset
    actors = torch_geometric.datasets.Actor(root="./real_data/actors")                          #nodes: 7600,  edges: 30019,  avg(degree): 07.90, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Actor.html#torch_geometric.datasets.Actor
    github = torch_geometric.datasets.GitHub(root="./real_data/github")                         #nodes: 37700, edges: 578006, avg(degree): 30.66, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GitHub.html#torch_geometric.datasets.GitHub

    airports_graph = torch_geometric.utils.convert.to_networkx(airports[0])
    wiki_graph = torch_geometric.utils.convert.to_networkx(wiki[0])
    facebook_graph = torch_geometric.utils.convert.to_networkx(facebook[0])
    actors_graph = torch_geometric.utils.convert.to_networkx(actors[0])
    github_graph = torch_geometric.utils.convert.to_networkx(github[0])

    sum_airports = 0
    for node_degree in airports_graph.degree:
        sum_airports += node_degree[1]
    sum_airports = sum_airports / len(airports_graph.nodes)

    sum_wiki = 0
    for node_degree in wiki_graph.degree:
        sum_wiki += node_degree[1]
    sum_wiki = sum_wiki / len(wiki_graph.nodes)

    sum_facebook = 0
    for node_degree in facebook_graph.degree:
        sum_facebook += node_degree[1]
    sum_facebook = sum_facebook / len(facebook_graph.nodes)

    sum_actors = 0
    for node_degree in actors_graph.degree:
        sum_actors += node_degree[1]
    sum_actors = sum_actors / len(actors_graph.nodes)

    sum_github = 0
    for node_degree in github_graph.degree:
        sum_github += node_degree[1]
    sum_github = sum_github / len(github_graph.nodes)

    a=0



if __name__ == "__main__":
    main()