import numpy as np
import random
import pandas as pd
import logging
import matplotlib.pyplot as plt

from numpy.core.fromnumeric import shape

# xzl: each @Data is for training, validation, testing, etc.
#   stored in column format
class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data

# xzl) load from files (saved by  preprosee_data.py) 
#   parse and return multi @Data sets ... this is the entry func. 
#   each Data stored in multiple columns (src,dest,timestamps..), time ordered.
def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name)) #xzl: dataframe from csv
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
    
  print("xzl: edge features shape {}".format(edge_features.shape))
  
  #edge_features = np.ones_like(edge_features)
  #logging.getLogger().info("xzl: overwrite features with 1s...")
  #print(edge_features[10])

  if randomize_features: # xzl) NB: wikipedia/reddit has not node features. all 0s
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  # xzl) split dataset by time
  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  # xzl) true/false masks?
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  # xzl) element wise logic and
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness      xzl) NB @timestamps is a Series. still elementwise AND
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  #xzl) use boolean array to select rows...
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  # xzl: for inductiveness?
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  interval_sources = dict()  #xzl
  src_edge_cnt = dict() # xzl
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
      src_edge_cnt[source_id] = 1 # xzl
    else:
      src_edge_cnt[source_id] += 1 # xzl      
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    # xzl
    if last_timestamp_sources[source_id] != 0:  # interacted before 
      if source_id not in interval_sources.keys():
        interval_sources[source_id] = []
      interval_sources[source_id].append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  '''
  # xzl: deal with src with only 1 interaction.... no good solution. we are done, last ts?   
  c_timestamp = timestamps[-1]
  for src, ts in last_timestamp_sources.items():
    #assert (ts != 0)
    #assert src not in interval_sources.keys()
    if src not in interval_sources.keys():
      interval_sources[src] = []
    interval_sources[src].append(c_timestamp-ts)
    #interval_sources[src] = [float("inf")] # only interaction once. max
  '''
  print("--- xzl")
  #interval_medians = dict()
  interval_medians = []
  intervals = []
  for s,i in interval_sources.items():
    #interval_medians[s] = np.median(i)
    interval_medians.append(np.median(i))
    intervals += i
  edge_cnt = []
  #for s,i in interval_sources.items():
  #  interval_cnt.append(len(i))
  for s,i in src_edge_cnt.items():
    edge_cnt.append(i)
  hist = np.histogram(edge_cnt, bins=20)
  print('hist for per-node edge cnt', hist)
  #print(np.histogram(edge_cnt, bins=[0,1,2,3,4,5,10,100,1000,10000]))
  plt.hist(edge_cnt, bins=20)
  plt.savefig("hist-edge-cnt.png")
    
  hist2 = np.histogram(intervals, bins=20)
  print("intervals for all nodes", hist2)

  hist2 = np.histogram(interval_medians, bins=20)
  print("median intervals for all nodes", hist2)
  
  mask = [x < hist[1][1] for x in src_edge_cnt]
  #print(mask)
  masked_medians=np.array(interval_medians)[mask]
  hist2 = np.histogram(masked_medians, bins=20)
  print("median intervals for nodes w/ fewer edges", hist2)

  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
