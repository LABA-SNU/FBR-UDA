import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

def distribution_check(dataloader):
  distribution = list()
  for a, b in tqdm(dataloader):
      input = a
      target = b
      for i in target:
          distribution.append(i.item())
  unique = set(distribution)

  dist = {}
  for i in range(len(unique)):
      cnt = distribution.count(i)
      dist[i] = cnt
      print(str(i) ,' : ', cnt)
  return dist

def calculate_class_weights(dataloader):
    class_dist = distribution_check(dataloader)
    class_weights = [sum(list(class_dist.values())) / (len(class_dist) * x) for x in list(class_dist.values())]
    class_weights = torch.FloatTensor(class_weights)
    return class_weights