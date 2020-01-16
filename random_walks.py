import numpy as np
import networkx as nx
import random
import pickle

def read_graph(path="./data/cn_assertions_filtered.tsv"):
  '''
  Reads the input network in networkx.
  '''

  G = nx.read_edgelist(path, nodetype=str, data=(('edge_type', str),), create_using=nx.DiGraph(), delimiter="\t")
  for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1
  return G


class Graph():
  def __init__(self, nx_G, is_directed, p, q):
    self.G = nx_G
    self.is_directed = is_directed
    self.p = p
    self.q = q

  def node2vec_walk(self, walk_length, start_node):
    '''
    Simulate a random walk starting from start node.
    '''
    G = self.G
    alias_nodes = self.alias_nodes
    alias_edges = self.alias_edges

    walk = [start_node]

    while len(walk) < walk_length:
      cur = walk[-1]
      cur_nbrs = sorted(G.neighbors(cur))
      if len(cur_nbrs) > 0:
        if len(walk) == 1:
          # TODO: This is Annes main change to the code, the rest is original node2vec code
          # NEW
          n = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
          walk.append(G.get_edge_data(cur, n)["edge_type"])
          walk.append(n)

          #walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
        else:
          #prev = walk[-2]
          prev = walk[-3]
          next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                     alias_edges[(prev, cur)][1])]
          ## new
          walk.append(G.get_edge_data(cur, next)["edge_type"])
          ####
          walk.append(next)
      else:
        break

    return walk

  def simulate_walks(self, num_walks, walk_length):
    '''
    Repeatedly simulate random walks from each node.
    '''
    G = self.G
    walks = []
    nodes = list(G.nodes())
    print
    'Walk iteration:'
    for walk_iter in range(num_walks):
      print
      str(walk_iter + 1), '/', str(num_walks)
      random.shuffle(nodes)
      for node in nodes:
        walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

    return walks

  def get_alias_edge(self, src, dst):
    '''
    Get the alias edge setup lists for a given edge.
    '''
    G = self.G
    p = self.p
    q = self.q

    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
      if dst_nbr == src:
        unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
      elif G.has_edge(dst_nbr, src):
        unnormalized_probs.append(G[dst][dst_nbr]['weight'])
      else:
        unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)

  def preprocess_transition_probs(self):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    G = self.G
    is_directed = self.is_directed

    alias_nodes = {}
    for node in G.nodes():
      unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
      norm_const = sum(unnormalized_probs)
      normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
      alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges = {}
    triads = {}

    if is_directed:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
    else:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

    self.alias_nodes = alias_nodes
    self.alias_edges = alias_edges

    return


def alias_setup(probs):
  '''
  Compute utility lists for non-uniform sampling from discrete distributions.
  Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
  for details
  '''
  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=np.int)

  smaller = []
  larger = []
  for kk, prob in enumerate(probs):
    q[kk] = K * prob
    if q[kk] < 1.0:
      smaller.append(kk)
    else:
      larger.append(kk)

  while len(smaller) > 0 and len(larger) > 0:
    small = smaller.pop()
    large = larger.pop()

    J[small] = large
    q[large] = q[large] + q[small] - 1.0
    if q[large] < 1.0:
      smaller.append(large)
    else:
      larger.append(large)

  return J, q


def alias_draw(J, q):
  '''
  Draw sample from a non-uniform discrete distribution using alias sampling.
  '''
  K = len(J)

  kk = int(np.floor(np.random.rand() * K))
  if np.random.rand() < q[kk]:
    return kk
  else:
    return J[kk]


"""
  parser.add_argument('--walk-length', type=int, default=80,
                      help='Length of walk per source. Default is 80.')

  parser.add_argument('--num-walks', type=int, default=10,
                      help='Number of walks per source. Default is 10.')

  parser.add_argument('--workers', type=int, default=8,
                      help='Number of parallel workers. Default is 8.')

  parser.add_argument('--p', type=float, default=1,
                      help='Return hyperparameter. Default is 1.')

  parser.add_argument('--q', type=float, default=1,
                      help='Inout hyperparameter. Default is 1.')

"""

def generate_random_walks_from_assertions():
  p = 1.0  # return hyperparameter
  q = 1.0  # inout hyperparameter
  is_directed = True  # whether the graph is directed
  num_walks = 2  # number of wandom walks per source def. 10
  walk_length = 15  # length of walk per source def. 80

  nx_G = read_graph(path="./data/cn_assertions_filtered.tsv")
  G = Graph(nx_G, is_directed, p, q)
  G.preprocess_transition_probs()
  walks = G.simulate_walks(num_walks, walk_length)
  filename = "./data/random_walk_" + str(p) + "_" + str(q) + "_" + str(num_walks) + "_" + str(walk_length) + ".p"
  with open(filename, 'wb') as handle:
    pickle.dump(walks, handle)
  print(len(walks))


def analyze_graph():
  nx_G = read_graph(path="./data/cn_assertions_filtered.tsv")
  print("%d nodes in the graph" % nx_G.number_of_nodes())
  print("%d edges in the graph" % nx_G.number_of_edges())
  print("%f density of graph" % nx.density(nx_G))
  #print("%f density of graph" % nx.number_of_selfloops(nx_G))
  print("%s" % nx.info(nx_G))
  print("%f avg in-degree" % float(float(sum(nx_G.in_degree().values()))/float(len(nx_G.in_degree().values()))))
  print("%f min in-degree" % float(float(min(nx_G.in_degree().values()))))
  print("%f max in-degree" % float(float(max(nx_G.in_degree().values()))))
  print("%f std in-degree" % float(float(np.std(np.array([float(v) for v in nx_G.in_degree().values()], dtype=np.float)))))
  print("%f avg in-degree" % float(float(np.average(np.array([float(v) for v in nx_G.in_degree().values()], dtype=np.float)))))

  print("%f avg out-degree" % float(float(sum(nx_G.out_degree().values()))/float(len(nx_G.out_degree().values()))))
  print("%f min out-degree" % float(float(min(nx_G.out_degree().values()))))
  print("%f max out-degree" % float(float(max(nx_G.out_degree().values()))))
  print("%f std out-degree" % float(float(np.std(np.array([float(v) for v in nx_G.out_degree().values()], dtype=np.float)))))
  print("%f avg out-degree" % float(float(np.average(np.array([float(v) for v in nx_G.out_degree().values()], dtype=np.float)))))


  comps_strong = list(nx.strongly_connected_component_subgraphs(nx_G))
  print("%d num strongly connected components" % len(comps_strong))
  comps_weak = list(nx.weakly_connected_component_subgraphs(nx_G))
  print("%d num weakly connected components" % len(comps_weak))
  diameters=[]
  for c in comps_strong:
    diameters.append(nx.diameter(c))
  print("Avg diameter %f for strongly connected components" % float(sum(diameters)/len(diameters)))
  print("Max diameter %f for strongly connected components" % max(diameters))
  print("Min diameter %f for strongly connected components" % min(diameters))
  print("%f std diameter" % float(float(np.std(np.array(diameters, dtype=np.float)))))
  print("%f avg diameter" % float(float(np.average(np.array(diameters, dtype=np.float)))))


def load_random_walk(p):
  walk = pickle.load(open(p, 'rb'))
  return walk


def main():
  #generate_random_walks_from_assertions()
  #analyze_graph()
  load_random_walk(p="./data/random_walk_1.0_1.0_2_10.p")

if __name__=="__main__":
  main()