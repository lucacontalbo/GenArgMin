import torch
import datasets
import pickle

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AdamW
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR, StepLR

from utils import set_random_seeds, get_config, get_device
from data_processor import StudentEssayProcessor, StudentEssayWithDiscourseInjectionProcessor,\
                            DebateProcessor, DebateWithDiscourseInjectionProcessor,\
                            MARGProcessor, MARGWithDiscourseInjectionProcessor,\
                            DiscourseMarkerProcessor, dataset,\
                            collate_fn, collate_fn_adv, collate_fn_graph
from batch_sampler import BalancedSampler
from models import AdversarialNet, BaselineModel, GraphModel
from train import Trainer

DATA_PATH = Path("data/")

def run():
  config = get_config()
  device = get_device()
  set_random_seeds(config["seed"])

  if config["dataset"] == "student_essay":
    if config["injection"]:
      processor = StudentEssayWithDiscourseInjectionProcessor(config, device)
    else:
      processor = StudentEssayProcessor(config, device)

    path_data = DATA_PATH / "student_essay.csv"
  elif config["dataset"] == "debate":
    if config["injection"]:
      processor = DebateWithDiscourseInjectionProcessor(config, device)
    else:
      processor = DebateProcessor(config, device)

    path_data = DATA_PATH / "debate.csv"
  elif config["dataset"] == "m-arg":
    if config["injection"]:
      processor = MARGWithDiscourseInjectionProcessor(config, device)
    else:
      processor = MARGProcessor(config, device)

    path_data = DATA_PATH / "presidential_final.csv"
  else:
    raise ValueError(f"{config['dataset']} is not a valid database name (choose between 'student_essay', 'debate' and 'm-arg')")

  data_train, data_dev, data_test = processor.read_input_files(path_data, name="train")

  if config["adversarial"]:
    df = datasets.load_dataset("discovery","discovery", trust_remote_code=True)
    adv_processor = DiscourseMarkerProcessor(config, device)
    if not config["dataset_from_saved"]:
      print("processing discourse marker dataset...")
      train_adv = adv_processor.process_dataset(df["train"])
      with open("./adv_dataset.pkl", "wb") as writer:
        pickle.dump(train_adv, writer)
    else:
      with open("./adv_dataset.pkl", "rb") as reader:
        train_adv = pickle.load(reader)

    data_train_tot = data_train + train_adv
  else:
    data_train_tot = data_train

  train_set = dataset(data_train_tot)
  dev_set = dataset(data_dev)
  test_set = dataset(data_test)

  if not config["adversarial"]:
    if not config["use_graph"]:
      model = BaselineModel(config)
      col_fn = collate_fn
    else:
      model = GraphModel(config)
      col_fn = collate_fn_graph
    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=col_fn)
  else:
    sampler_train = BalancedSampler(data_train, train_adv, config["batch_size"])
    train_dataloader = DataLoader(train_set, batch_sampler=sampler_train, collate_fn=collate_fn)

    model = AdversarialNet(config)

    if len(config["visualize"]) > 0:
        try:
            model.load_state_dict(torch.load(f"./{config['dataset']}_model.pt"))
            model.eval()
        except:
          raise FileNotFoundError(f"Model \"./{config['dataset']}_model.pt\" does not exist. Train the model first, then you can visualize the embeddings")

  model.to(device)

  dev_dataloader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
  test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": 0.01,
    },
    {
      "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      "weight_decay": 0.0
    },
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"], weight_decay=config["weight_decay"])

  if config["scheduler"]:
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=30)

  loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(config["class_weight"]).to(device))

  best_dev_f1 = -1
  result_metrics = []

  trainer = Trainer(config, device)

  if len(config["visualize"]) > 0:
    trainer.visualize(model, test_dataloader, config)
  elif config["grid_search"]:
    range_disc = np.arange(0,1.2,0.2)
    range_adv = np.arange(0,1.2,0.2)

    for discovery_weight in range_disc:
      for adv_weight in range_adv:
        for epoch in range(config["epochs"]):
          print('===== Start training: epoch {}, lr {} ====='.format(epoch + 1, scheduler.get_lr()))
          print(f"*** trying with discovery_weight = {discovery_weight}, adv_weight = {adv_weight}")
          trainer.train(epoch, model, loss_fn, optimizer, train_dataloader, discovery_weight=discovery_weight, adv_weight=adv_weight, scheduler=scheduler)
          dev_a, dev_p, dev_r, dev_f1 = trainer.val(model, dev_dataloader)
          test_a, test_p, test_r, test_f1 = trainer.val(model, test_dataloader)
          if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_acc, best_test_pre, best_test_rec, best_test_f1 = test_a, test_p, test_r, test_f1
            torch.save(model.state_dict(), f"./{config['dataset']}_model.pt")

        print('*** best result on test set ***')
        print(best_test_acc)
        print(best_test_pre)
        print(best_test_rec)
        print(best_test_f1, end="\n")

        result_metrics.append([best_test_acc, best_test_pre, best_test_rec, best_test_f1])

        #we reset the model and optimizer in order to start from the same random seed
        #this makes the results reproducible even without running gridsearch

        del model
        del optimizer

        set_random_seeds(config["seed"])
        
        model = AdversarialNet(config)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
        if config["scheduler"]:
           scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=30)

        best_dev_f1 = -1
  else:
    for epoch in range(config["epochs"]):
      if config["scheduler"]:
        print('===== Start training: epoch {}, lr {} ====='.format(epoch + 1, scheduler.get_lr()))
        trainer.train(epoch, model, loss_fn, optimizer, train_dataloader,
                      discovery_weight=config["discovery_weight"], adv_weight=config["adv_weight"], scheduler=scheduler)
      else:
        print('===== Start training: epoch {} ====='.format(epoch + 1))
        trainer.train(epoch, model, loss_fn, optimizer, train_dataloader,
                      discovery_weight=config["discovery_weight"], adv_weight=config["adv_weight"])

      dev_a, dev_p, dev_r, dev_f1 = trainer.val(model, dev_dataloader)
      test_a, test_p, test_r, test_f1 = trainer.val(model, test_dataloader)
      if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_test_acc, best_test_pre, best_test_rec, best_test_f1 = test_a, test_p, test_r, test_f1
        torch.save(model.state_dict(), f"./{config['dataset']}_model.pt")

    print('*** best result on test set ***')
    print(best_test_acc)
    print(best_test_pre)
    print(best_test_rec)
    print(best_test_f1, end="\n")
    result_metrics.append([best_test_acc, best_test_pre, best_test_rec, best_test_f1])

  print(f"Overall metrics: {result_metrics}")

if __name__ == "__main__":
  run()