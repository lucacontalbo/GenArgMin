import torch

import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from utils import get_device, get_json

JSON_PATH = Path("json/")
PRETRAINED_EMB_PATHS = Path("pretrained_embs/")

class dataset(Dataset):
    def __init__(self, examples):
        super(dataset, self).__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

def collate_fn(examples):
    ids_sent1, segs_sent1, att_mask_sent1, labels = map(list, zip(*examples))

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return ids_sent1, segs_sent1, att_mask_sent1, labels

def collate_fn_graph(examples):
    ids_sent1, segs_sent1, att_mask_sent1, sentence1, sentence2, labels = map(list, zip(*examples))

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return ids_sent1, segs_sent1, att_mask_sent1, sentence1, sentence2, labels

def collate_fn_adv(examples):
    ids_sent1, segs_sent1, att_mask_sent1, labels = map(list, zip(*examples))

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)

    return ids_sent1, segs_sent1, att_mask_sent1, labels


class DataProcessor:

  def __init__(self,config, device):
    self.config = config
    self.device = device
    self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
    self.max_sent_len = config["max_sent_len"]
    self.graphrelation2words = get_json(JSON_PATH / "graphrelation2words.json")

  def __str__(self,):
    pattern = """General data processor: \n\n Tokenizer: {}\n\nMax sentence length: {}""".format(self.config["model_name"], self.max_sent_len)
    return pattern

  def _get_examples(self, dataset, dataset_type="train"):
    examples = []
    ids_sent, segs_sent = [], []
    max_length = 0
    for row in tqdm(dataset, desc="tokenizing..."):
      _, sentence1, sentence2, label = row

      sentence1_length = len(self.tokenizer.encode(sentence1))
      sentence2_length = len(self.tokenizer.encode(sentence2))

      ids_sent1 = self.tokenizer.encode(sentence1, sentence2)
      segs_sent1 = [0] * sentence1_length + [1] * (sentence2_length)

      assert len(ids_sent1) == len(segs_sent1)

      ids_sent.append(ids_sent1)
      segs_sent.append(segs_sent1)

      if max_length < len(ids_sent1):
        max_length = len(ids_sent1)

    pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
    print(f"Maximum length of sequence: {max_length}")
    self.max_sent_len = max_length

    for ids_sent1, segs_sent1 in zip(ids_sent, segs_sent):
      if len(ids_sent1) <= self.max_sent_len:        
        res = self.max_sent_len - len(ids_sent1)
        att_mask_sent1 = [1] * len(ids_sent1) + [0] * res
        ids_sent1 += [pad_id] * res
        segs_sent1 += [0] * res
      else:
        print(f"Tokens length of {len(ids_sent1)} is higher than maximum length {self.max_sent_len}")
        ids_sent1 = ids_sent1[:self.max_sent_len]
        segs_sent1 = segs_sent1[:self.max_sent_len]
        att_mask_sent1 = [1] * self.max_sent_len

      example = [ids_sent1, segs_sent1, att_mask_sent1, sentence1, sentence2, label]

      examples.append(example)

    print(f"finished preprocessing examples in {dataset_type}")

    return examples


class DiscourseMarkerProcessor(DataProcessor):

  def __init__(self, config, device):
    super(DiscourseMarkerProcessor, self).__init__(config, device)

    self.mapping = get_json(JSON_PATH / "word_to_target.json")
    self.id_to_word = get_json(JSON_PATH / "id_to_word.json")

  def process_dataset(self, dataset, name="train"):
    result = []
    new_dataset = []

    for sample in dataset:
      if self.id_to_word[str(sample["label"])] not in self.mapping.keys():
        continue

      new_dataset.append([sample["sentence1"], sample["sentence2"], self.mapping[self.id_to_word[str(sample["label"])]]])

    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    labels = []

    for i, sample in tqdm(enumerate(new_dataset), desc="processing labels..."):
      labels.append([sample[-1]])

    print("one hot encoding...")
    labels = one_hot_encoder.fit_transform(labels)

    for i, (sample, label) in tqdm(enumerate(zip(new_dataset, labels)), desc="creating results..."):
      result.append([f"{name}_{i}", sample[0], sample[1], [], label])

    examples = self._get_examples(result, name)
    return examples


class StudentEssayProcessor(DataProcessor):

  def __init__(self, config, device):
    super(StudentEssayProcessor,self).__init__(config, device)


  def read_input_files(self, file_path, name="train", pipe=None):

      result_train = []
      result_dev = []
      result_test = []

      df = pd.read_csv(file_path, index_col=0)
      for i,row in tqdm(df.iterrows()):
        sample_id = row.iloc[0]
        sent = row.iloc[1].strip()
        target = row.iloc[2].strip()
        if pipe is not None:
          ds_marker = pipe(f"{sent}</s></s>{target}")[0]["label"]
          ds_marker = ds_marker.replace("_", " ")
          ds_marker = ds_marker[0].upper() + ds_marker[1:]
          target = target[0].lower() + target[1:]
          target = ds_marker + " " + target

        label = row.iloc[4]
        split = row.iloc[5]

        if not label:
          l = [1,0]
        else:
          l = [0,1]
              
        if split == "train":
          result_train.append([sample_id, sent, target, l])
        elif split == "dev":
          result_dev.append([sample_id, sent, target, l])
        elif split == "test":
          result_test.append([sample_id, sent, target, l])
        else:
          raise ValueError(f"unknown dataset split: {split}")

      examples_train = self._get_examples(result_train, name)
      examples_dev = self._get_examples(result_dev, name)
      examples_test = self._get_examples(result_test, name)

      return examples_train, examples_dev, examples_test


class DebateProcessor(DataProcessor):

  def __init__(self, config, device):
    super(DebateProcessor,self).__init__(config, device)

  def read_input_files(self, file_path, name="train", pipe=None):
      result_train = []
      result_dev = []
      result_test = []

      df = pd.read_csv(file_path, index_col=0)
      for i,row in df.iterrows():
        sample_id = row.iloc[0]
        sent = row.iloc[1].strip()
        target = row.iloc[2].strip()

        label = row.iloc[4]
        split = row.iloc[5]

        if pipe is not None:
          ds_marker = pipe(f"{sent}</s></s>{target}")[0]["label"]
          ds_marker = ds_marker.replace("_", " ")
          ds_marker = ds_marker[0].upper() + ds_marker[1:]
          target = target[0].lower() + target[1:]
          target = ds_marker + " " + target

        l = [0,0]
        if not label:
          l = [1,0]
        else:
          l = [0,1]

        if split == "train":
          result_train.append([sample_id, sent, target, l])
        elif split == "dev":
          result_dev.append([sample_id, sent, target, l])
        elif split == "test":
          result_test.append([sample_id, sent, target, l])
        else:
          raise ValueError(f"unknown dataset split: {split}")

      examples_train = self._get_examples(result_train, name)
      examples_dev = self._get_examples(result_dev, name)
      examples_test = self._get_examples(result_test, name)

      return examples_train, examples_dev, examples_test


class MARGProcessor(DataProcessor):

  def __init__(self, config, device):
    super(MARGProcessor, self).__init__(config, device)

  def read_input_files(self, file_path, name="train", pipe=None):

      result_train = []
      result_dev = []
      result_test = []

      df = pd.read_csv(file_path)
      for i,row in tqdm(df.iterrows()):
              sample_id = row.iloc[0]
              sent = row.iloc[1].strip()
              target = row.iloc[2].strip()

              if pipe is not None:
                ds_marker = pipe(f"{sent}</s></s>{target}")[0]["label"]
                ds_marker = ds_marker.replace("_", " ")
                ds_marker = ds_marker[0].upper() + ds_marker[1:]
                target = target[0].lower() + target[1:]
                target = ds_marker + " " + target

              label = row.iloc[3].strip()
              split = row.iloc[-1]

              l=[0,0,0]
              if label == 'support':
                l = [1,0,0]
              elif label == 'attack':
                l = [0,1,0]
              elif label == 'neither':
                l = [0,0,1]

              if split == "train":
                result_train.append([sample_id, sent, target, l])
              elif split == "dev":
                result_dev.append([sample_id, sent, target, l])
              elif split == "test":
                result_test.append([sample_id, sent, target, l])
              else:
                raise ValueError(f"unknown dataset split: {split}")

      examples_train = self._get_examples(result_train, name)
      examples_dev = self._get_examples(result_dev, name)
      examples_test = self._get_examples(result_test, name)

      return examples_train, examples_dev, examples_test


class StudentEssayWithDiscourseInjectionProcessor(StudentEssayProcessor):

  def __init__(self, config, device):
    super(StudentEssayWithDiscourseInjectionProcessor, self).__init__(config, device)
    self.pipe = pipeline("text-classification", model=config["model_name"])

  def read_input_files(self, file_path, name="train"):
      examples = super().read_input_files(file_path, name, pipe=self.pipe)
      return examples


class DebateWithDiscourseInjectionProcessor(DebateProcessor):

  def __init__(self, config, device):
    super(DebateWithDiscourseInjectionProcessor,self).__init__(config, device)
    self.pipe = pipeline("text-classification", model=config["model_name"])

  def read_input_files(self, file_path, name="train"):
      examples = super().read_input_files(file_path, name, pipe=self.pipe)
      return examples


class MARGWithDiscourseInjectionProcessor(DataProcessor):

  def __init__(self, config, device):
    super(MARGWithDiscourseInjectionProcessor,self).__init__(config, device)
    self.pipe = pipeline("text-classification", model=config["model_name"])

  def read_input_files(self, file_path, name="train"):
      examples = super().read_input_files(file_path, name, pipe=self.pipe)
      return examples
