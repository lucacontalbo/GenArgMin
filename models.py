import torch
import spacy
import gc
import string
import networkx
import torch_geometric as tg
import matplotlib
import matplotlib.pyplot

from torch_geometric.data import HeteroData, Batch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

spacy.load('en_core_web_sm')

class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(bkw, x, lambda_value=0.01):
        bkw.lambda_value = torch.tensor(lambda_value)
        return x.reshape_as(x)

    @staticmethod
    def backward(bkw, prev_gradient):
        post_gradient = prev_gradient.clone()
        return bkw.lambda_value * post_gradient.neg(), None


class AdversarialNet(torch.nn.Module):
  def __init__(self, config):
    super(AdversarialNet, self).__init__()

    self.num_classes = config["num_classes"]
    self.num_classes_adv = config["num_classes_adv"]
    self.embed_size = config["embed_size"]
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.linear_layer_adv = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes_adv)
    self.task_linear = torch.nn.Linear(in_features=self.embed_size, out_features=2)

    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self._init_weights(self.linear_layer)
    self._init_weights(self.linear_layer_adv)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)
    self._init_weights(self.task_linear)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    tar_mask_sent1 = (segs_sent1 == 0).long()
    tar_mask_sent2 = (segs_sent1 == 1).long()

    H_sent1 = torch.mul(tar_mask_sent1.unsqueeze(2), embed_sent1)
    H_sent2 = torch.mul(tar_mask_sent2.unsqueeze(2), embed_sent1)

    K_sent1 = self.K(H_sent1)
    V_sent1 = self.V(H_sent1)
    Q_sent2 = self.Q(H_sent2)

    att_output = self.multi_head_att(Q_sent2, K_sent1, V_sent1)

    H_sent = torch.mean(att_output[0], dim=1)

    if visualize:
      return H_sent
    
    if self.training:
      batch_size = H_sent.shape[0]
      samples = H_sent[:batch_size // 2, :]
      samples_adv = H_sent[batch_size // 2:, ]

      predictions = self.linear_layer(samples)
      predictions_adv = self.linear_layer_adv(samples_adv)

      mean_grl = GRLayer.apply(torch.mean(embed_sent1, dim=1), .01)
      task_prediction = self.task_linear(mean_grl)

      return predictions, predictions_adv, task_prediction
    else:
      predictions = self.linear_layer(H_sent)

      return predictions


class BaselineModel(torch.nn.Module):
  def __init__(self, config):
    super(BaselineModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]
    self.first_last_avg = config["first_last_avg"]

    self.plm = AutoModel.from_pretrained(config["model_name"])
    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self._init_weights(self.linear_layer)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.multi_head_att)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    tar_mask_sent1 = (segs_sent1 == 0).long()
    tar_mask_sent2 = (segs_sent1 == 1).long()

    H_sent1 = torch.mul(tar_mask_sent1.unsqueeze(2), embed_sent1)
    H_sent2 = torch.mul(tar_mask_sent2.unsqueeze(2), embed_sent1)

    K_sent1 = self.K(H_sent1)
    V_sent1 = self.V(H_sent1)
    Q_sent2 = self.Q(H_sent2)

    att_output = self.multi_head_att(Q_sent2, K_sent1, V_sent1)

    H_sent = torch.mean(att_output[0], dim=1)

    if visualize:
      return H_sent

    predictions = self.linear_layer(H_sent)

    return predictions

class GraphModel(torch.nn.Module):
  
  def __init__(self, config, gen_steps=2):

    super(GraphModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]
    self.first_last_avg = config["first_last_avg"]
    self.batch_size = config["batch_size"]
    self.generated_max_length = config["generated_max_length"]
    self.k = config["k"]

    self.graph_tokenizer = AutoTokenizer.from_pretrained("mismayil/comet-bart-ai2")
    self.graph_model = AutoModelForSeq2SeqLM.from_pretrained("mismayil/comet-bart-ai2").to("cuda")
    self.rels = ["ObjectUse", "AtLocation", "MadeUpOf", "HasProperty", "CapableOf", "Desires", "NotDesires", "IsAfter",
                 "HasSubEvent", "IsBefore", "HinderedBy", "Causes", "xReason", "isFilledBy", "xNeed", "xAttr", "xEffect",
                 "xReact", "xWant", "xIntent", "oEffect", "oReact", "oWant"]
    self.gen_steps = gen_steps

    self.nlp = spacy.load("en_core_web_sm")

    self.plm = AutoModel.from_pretrained(config["model_name"]).to("cuda")
    self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    self.rels_embeddings = self.get_knowledge_embedding_bert(self.rels)
    self.related_to_embedding = self.get_knowledge_embedding_bert(["RelatedTo"])

    config = self.plm.config
    config.type_vocab_size = 4
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)

    for param in self.plm.parameters():
      param.requires_grad = True
    for param in self.graph_model.parameters():
      param.requires_grad = False

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    self.Q1 = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K1 = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.V1 = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    graph_metadata = (
      ["sent1", "sent2", "csk", "total"],
      [
        ("sent1", "edge", "csk"),
        ("csk", "edge", "csk"),
        ("sent2", "edge", "csk"),
        ("sent1", "edge_total", "total"),
        ("sent2", "edge_total", "total"),
        ("csk", "edge_total", "total"),
        ("sent1", "self_loop", "sent1"),
        ("sent2", "self_loop", "sent2"),
        ("csk", "self_loop", "csk"),
        ("total", "self_loop", "total")
      ]
    )

    self.gnn_layers = torch.nn.ModuleList([
      tg.nn.HGTConv(self.embed_size, self.embed_size, graph_metadata),
      tg.nn.HGTConv(self.embed_size, self.embed_size, graph_metadata)
    ])

    self.graph_as_dict = [([],[]) for _ in range(self.batch_size)]

    self._init_weights(self.linear_layer)
    self._init_weights(self.Q)
    self._init_weights(self.K)
    self._init_weights(self.V)
    self._init_weights(self.Q1)
    self._init_weights(self.K1)
    self._init_weights(self.V1)
    self._init_weights(self.multi_head_att)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
  
  def set_node_dict(self):
    return [{} for _ in range(self.batch_size)]

  def get_initial_text_spans(self, sents):
    spans = []
    for sent in sents:
      spans.append([sent + " {} [GEN]"])

    docs = self.nlp.pipe(sents)
    for i, doc in enumerate(docs):
      subjects = [token.text+" {} [GEN]" for token in doc if token.dep_ in ("nsubj", "nsubjpass")]
      objects = [token.text+" {} [GEN]" for token in doc if token.dep_ in ("obj", "iobj", "dobj", "pobj")]

      spans[i].extend(subjects)
      spans[i].extend(objects)

    return spans

  def get_documents(self, query, ngram_range, reverse_order=True):
    documents = []
    query = query[1:-1]
    if reverse_order:
      span_sizes = [el for el in range(ngram_range[1], ngram_range[0]-1, -1)] 
    else:
      span_sizes = [el for el in range(ngram_range[0], ngram_range[1]+1)] 

    for span_size in span_sizes:
      for i in range(len(query)-span_size+1):
        embedding = torch.mean(query[i:i+span_size,:], dim=0).unsqueeze(0)
        documents.append([embedding, (i+1,i+span_size+1)])

    return documents


  def mmr(self, query, ngram_range, diversity, lemmatize=True, remove_stopwords=False, query_str=None):
    num_docs_to_return = self.k
    if (remove_stopwords or lemmatize) and not isinstance(query_str, str):
      raise ValueError("you must provide query_str when removing stopwords or lemmatization")

    if remove_stopwords:
      eng_stopwords = stopwords.words('english')
    else:
      eng_stopwords = []

    tokenized_query_str = self.tokenizer.encode(query_str, add_special_tokens=True)
    #query_str = query_str.lower()

    not_stopwords_idx = [i for i in range(1, len(tokenized_query_str)-1) if self.tokenizer.decode(tokenized_query_str[i]).lower().strip() not in eng_stopwords]
    non_punct_idx = [i for i in range(1, len(tokenized_query_str)-1) if self.tokenizer.decode(tokenized_query_str[i]).strip() not in string.punctuation]
    not_stopwords_idx = list(set(not_stopwords_idx).intersection(set(non_punct_idx)))

    query = query[[0] + not_stopwords_idx + [len(tokenized_query_str)-1],:]
    query_str = self.tokenizer.decode([tokenized_query_str[el] for el in not_stopwords_idx])

    documents = self.get_documents(query, ngram_range)
    query = torch.mean(query, dim=0)
    selected_docs = []

    if len(query.shape) == 1:
      query = query.unsqueeze(0)

    doc_embeddings = torch.cat([doc[0] if len(doc[0].shape) != 1 else doc[0] for doc in documents], dim=0)
    doc2query_similarity = torch.from_numpy(cosine_similarity(doc_embeddings.cpu().detach().numpy(), query.cpu().detach().numpy()).squeeze(1))
    doc2doc_similarity = torch.from_numpy(cosine_similarity(doc_embeddings.cpu().detach().numpy()))

    keywords_idx = [torch.argmax(doc2query_similarity)]
    candidates_idx = [i for i in range(len(doc2query_similarity)) if i != keywords_idx[0]]

    for _ in range(1,min(num_docs_to_return, len(documents))):
      candidate_similarities = doc2query_similarity[candidates_idx]
      target_similarities, _ = torch.max(doc2doc_similarity[candidates_idx][:, keywords_idx], axis=1)

      mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities
      mmr_idx = candidates_idx[torch.argmax(mmr)]

      keywords_idx.append(mmr_idx)
      candidates_idx.remove(mmr_idx)

    for idx in keywords_idx:
      doc = documents[idx]
      doc.append(round(doc2query_similarity[idx].item(),4))
      doc.append(query_str)
      doc.append(self.tokenizer.decode([tokenized_query_str[i] for i in [0] + not_stopwords_idx + [len(tokenized_query_str)-1]][doc[1][0]:doc[1][1]]))

      selected_docs.append(doc)

    ordered_docs = sorted(selected_docs, key=lambda x: x[2], reverse=True)

    return ordered_docs

  def get_initial_key_claims(self, sents):
    spans = []
    for sent in sents:
      spans.append([sent + " {} [GEN]"])

    docs = self.nlp.pipe(sents)
    for i, doc in enumerate(docs):
      subjects = [token.text+" {} [GEN]" for token in doc if token.dep_ in ("nsubj", "nsubjpass")]
      objects = [token.text+" {} [GEN]" for token in doc if token.dep_ in ("obj", "iobj", "dobj", "pobj")]

      spans[i].extend(subjects)
      spans[i].extend(objects)

    return spans

  @torch.autocast(device_type="cuda")
  def generate_knowledge(self, sents):
    sents_with_preds = []
    sample_boundaries = []
    preds = []
    edge_index = []
    for sent in sents:
      last_el = sample_boundaries[-1] if len(sample_boundaries) > 0 else 0
      sample_boundaries.append(last_el + len(sent)*len(self.rels))
      for span in sent:
        for rel in self.rels:
          preds.append(rel)
          sents_with_preds.append(span.strip() + f" {rel} [GEN]")

    inputs = self.graph_tokenizer(sents_with_preds, return_tensors="pt", padding=True, truncation=True).to("cuda")
    num_return_sequences=5
    with torch.no_grad():
      outputs = self.graph_model.generate(
        inputs["input_ids"],
        max_length=10,
        num_beams=num_return_sequences,
        num_return_sequences=num_return_sequences,
        #early_stopping=True,
        return_dict_in_generate=True,
        output_hidden_states=True
      )

    generated_text = self.graph_tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
    new_outputs = torch.tensor([]).to("cuda")
    new_generated_texts = []
    for i in range(0,len(outputs["sequences"]),num_return_sequences):
      found = False
      for j in range(num_return_sequences):
        if len(generated_text[i+j].strip()) == 0:
          continue
        if generated_text[i+j].strip() == "none":
          continue
        if generated_text[i+j].strip() == ' '.join(sents_with_preds[i//num_return_sequences].split()[:-2]).strip():
          continue

        new_outputs = torch.cat([new_outputs, outputs["sequences"][i+j,:].unsqueeze(0)], dim=0)
        new_generated_texts.append(generated_text[i+j])
        found=True
        break

      if not found:
        new_outputs = torch.cat([new_outputs, outputs["sequences"][i,:].unsqueeze(0)], dim=0)
        new_generated_texts.append(generated_text[i])

    return new_outputs, new_generated_texts, sample_boundaries

  def get_knowledge_embedding_bert(self, outputs):
    tok_sents = self.tokenizer(outputs, padding='max_length', truncation=True, max_length=10, add_special_tokens=True)
    out_sent1 = self.plm(torch.tensor(tok_sents["input_ids"]).to("cuda"), attention_mask=torch.tensor(tok_sents["attention_mask"]).to("cuda"))
    return out_sent1.last_hidden_state

  def get_knowledge_embedding(self, outputs, embed_type="all_tokens"):
    new_outputs = torch.tensor([], dtype=torch.float32).to("cuda") # to be changed on var
    attn_mask = torch.tensor([], dtype=torch.float32).to("cuda")
    if embed_type == "all_tokens":
      for hidden_states, sequence in zip(outputs["decoder_hidden_states"], torch.permute(outputs["sequences"][:,1:], (1,0))):
        # we mask padding tokens in generated sequence
        sequence = (sequence != 1).long().unsqueeze(-1).expand(len(sequence), 1024).unsqueeze(1)
        attn_mask = torch.cat([attn_mask, sequence], dim=1)
        new_outputs = torch.cat([new_outputs, hidden_states[12]*sequence], dim=1)
        #del sequence
    
    return new_outputs, attn_mask

  def group_batches(self, outputs, attn_mask, sample_boundaries):
    output = torch.zeros(len(sample_boundaries), self.generated_max_length, 1024).to("cuda")
    attn = torch.zeros(len(sample_boundaries), self.generated_max_length, 1024).to("cuda")
    start_index = 0

    #assert len(sample_boundaries) == self.batch_size, f"sample_boundaries' length {len(sample_boundaries)} is not equal to the batch size {self.batch_size}"

    for i in range(len(sample_boundaries)):
      max_index = sample_boundaries[i]
      if max_index > 0:
        concatenated = outputs[start_index:max_index].reshape(-1, 1024)
        concatenated_attn = attn_mask[start_index:max_index].reshape(-1,1024)
        
        length = concatenated.shape[0]
        if length >= self.generated_max_length:
          concatenated = concatenated[:self.generated_max_length]
          concatenated_attn = concatenated_attn[:self.generated_max_length]
        elif length < self.generated_max_length:
          padding = torch.zeros(self.generated_max_length - length, 1024).to("cuda")
          concatenated = torch.cat((concatenated, padding), dim=0)
          concatenated_attn = torch.cat((concatenated_attn, padding), dim=0)

        output[i] = concatenated
        attn[i] = concatenated_attn
        start_index += (max_index - start_index)

    return output, attn

  def prune(self, texts, attn_weights):
    attn_weights_flat = torch.max(attn_weights,dim=2)[0].reshape(attn_weights.shape[0], -1)
    _, topk_indices = torch.topk(attn_weights_flat, self.k, dim=1)
    row_indices = topk_indices #// attn_weights.shape[2]

    topk_sentences = [[] for _ in range(self.batch_size)]
    err = 0
    for i in range(len(row_indices)):
      for j in range(len(row_indices[i])):
        try:
          topk_sentences[i].append(texts[i*attn_weights.shape[1] + row_indices[i][j]])
        except:
          err += 1

    assert err == 0, f"{err} wrong indices in pruning"
    return topk_sentences, row_indices

  def add_edges_to_graph(self, heterodata, edge_txt, edge_features, edge_index):
    #for txt in edge_txt:
    if not hasattr(heterodata[*edge_txt], "edge_index"):
      heterodata[*edge_txt].edge_index = torch.tensor([], dtype=torch.int32).to("cuda")
    if not hasattr(heterodata[*edge_txt], "edge_attr"):
      heterodata[*edge_txt].edge_attr = torch.tensor([]).to("cuda")
    
    edge_index = edge_index.type(torch.int32)

    heterodata[*edge_txt].edge_index = torch.cat([heterodata[*edge_txt].edge_index, edge_index.to("cuda")], dim=1)
    heterodata[*edge_txt].edge_attr = torch.cat([heterodata[*edge_txt].edge_attr, edge_features], dim=0)
    
    return heterodata

  def add_to_graph(self, heterodata, node_txt, node_features, sample_num=0, node_texts=[], add_self_loop=True):
    node_ids = []
    for node_feat, node_text in zip(node_features, node_texts):
      if node_txt not in self.node_dict[sample_num].keys():
        self.node_dict[sample_num][node_txt] = {}
      if node_text in self.node_dict[sample_num][node_txt].keys():
        continue

      if not hasattr(heterodata[node_txt], 'x'):
        heterodata[node_txt].x = torch.tensor([]).to("cuda")
      heterodata[node_txt].x = torch.cat([heterodata[node_txt].x, node_feat.unsqueeze(0).to("cuda")], dim=0)
      self.node_dict[sample_num][node_txt][node_text] = len(heterodata[node_txt].x)
      node_ids.append(len(heterodata[node_txt].x))

    if add_self_loop and len(node_ids) > 0:
      edge_txt = (node_txt, "self_loop", node_txt)
      edge_index = torch.transpose(torch.tensor([[k-1, k-1] for k in node_ids]),0,1)
      edge_features = torch.ones(self.k,self.embed_size).to("cuda") #TODO: change to custom edge features
      heterodata = self.add_edges_to_graph(heterodata, edge_txt, edge_features, edge_index)

    return heterodata

  def get_mmr(self,H_sent1, H_sent2, segs_sent, sent1, sent2):
    sent1_spans = []
    sent2_spans = []

    for sample1, sample2, ss1, s1, s2 in zip(H_sent1, H_sent2, segs_sent, sent1, sent2):
      first_one = torch.argmax(torch.flip(torch.arange(len(ss1)).to("cuda"), dims=[0]) * ss1)
      last_one = torch.argmax(torch.arange(len(ss1)).to("cuda") * ss1)
      mmr1 = self.mmr(sample1[:first_one,:], ngram_range=(2,4), diversity=0.4, remove_stopwords=False, query_str=s1)
      mmr2 = self.mmr(sample2[first_one:last_one+1,:], ngram_range=(2,4), diversity=0.4, remove_stopwords=False, query_str=s2)
      sent1_spans.append(mmr1)
      sent2_spans.append(mmr2)
    
    return sent1_spans, sent2_spans

  def create_edge_data(self, text_spans, link_type, sample_num):
    source_type, rel_type, dest_type = link_type

    edge_types = (source_type, rel_type, dest_type)
    edge_index = torch.transpose(
      torch.tensor([
        [
          self.node_dict[sample_num][source_type][text_spans[sample_num][k]]-1, 
          self.node_dict[sample_num][dest_type][dest_type]-1
        ] for k in range(len(text_spans[sample_num]))
      ]),
    0,1)

    edge_features = self.related_to_embedding[:,0,:].repeat(edge_index.shape[1],1), \      

    return edge_types, edge_features, edge_index


  def add_mmr_nodes_to_graph(self, heterodata, sent_knowl_emb, text_spans):

    for i in range(len(sent_knowl_emb[0])):
      for node_type, emb, text_span in zip(["sent1", "sent2"], sent_knowl_emb, text_spans):
        heterodata[i] = self.add_to_graph(heterodata[i], node_type, emb[i,0,:], sample_num=i, node_texts=text_span[i], add_self_loop=True)

        """for graph_as_dict, text_span in zip(self.graph_as_dict[i], [text_spans[0][i], text_spans[1][i]]):
          for t in text_span:
            graph_as_dict.append((t,))"""

        rnd_initialization = torch.normal(mean=torch.tensor([0.0 for _ in range(self.embed_size)]), std=torch.tensor([0.01 for _ in range(self.embed_size)]))
        heterodata[i] = self.add_to_graph(heterodata[i], "total", rnd_initialization.unsqueeze(0), sample_num=i, node_texts=["total" for _ in range(self.batch_size)])

        edge_types, edge_features, edge_index = self.create_edge_data(text_span, (node_type, "edge_total", "total"), i)

        heterodata[i] = self.add_edges_to_graph(heterodata[i], edge_types, edge_features, edge_index)
    
    return heterodata

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1, sent1, sent2, visualize=False):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)

    last_sent1, first_sent1 = out_sent1.hidden_states[-1], out_sent1.hidden_states[1]

    if self.first_last_avg:
      embed_sent1 = torch.div((last_sent1 + first_sent1), 2)
    else:
      embed_sent1 = last_sent1

    tar_mask_sent1 = (segs_sent1 == 0).long()
    tar_mask_sent2 = (segs_sent1 == 1).long()

    H_sent1 = torch.mul(tar_mask_sent1.unsqueeze(2), embed_sent1)
    H_sent2 = torch.mul(tar_mask_sent2.unsqueeze(2), embed_sent1)

    sent1_spans, sent2_spans = self.get_mmr(H_sent1, H_sent2, segs_sent1, sent1, sent2)

    heterodata = [HeteroData() for _ in range(self.batch_size)]
    self.node_dict = self.set_node_dict()
    k_1, v_1 = self.K(H_sent1), self.V(H_sent1)
    k_2, v_2 = self.K(H_sent2), self.V(H_sent2)

    text_spans1 = [[ex[-1] for ex in el] for el in sent1_spans]
    text_spans2 = [[ex[-1] for ex in el] for el in sent2_spans]

    sent1_knowl_emb = self.get_knowledge_embedding_bert([
      el for sample in text_spans1 for el in sample
    ]).reshape(self.batch_size,self.k,-1,self.embed_size)

    sent2_knowl_emb = self.get_knowledge_embedding_bert([
      el for sample in text_spans2 for el in sample
    ]).reshape(self.batch_size,self.k,-1,self.embed_size)
    
    heterodata = self.add_mmr_nodes_to_graph(
      heterodata,
      [sent1_knowl_emb,sent2_knowl_emb],
      [text_spans1,text_spans2]
    )

    text_spans = [text_spans1, text_spans2]
    for i in range(self.gen_steps):
      for k, (text_span, k_emb, v_emb, node_type) in enumerate(zip(text_spans, [k_2, k_1], [v_2, v_1], ["sent1", "sent2"])):
        sent_outputs, sent_gen_text, sample_boundaries = self.generate_knowledge(text_span)
        sent_knowl_emb = self.get_knowledge_embedding_bert(sent_gen_text)
        sent_knowl_emb = sent_knowl_emb.unsqueeze(0).reshape(self.batch_size, len(sent_gen_text) // self.batch_size,-1, self.embed_size)[:,:,0,:]

        q_emb = self.Q(sent_knowl_emb)
        sent_knowl_emb, attn_weights = self.multi_head_att(q_emb, k_emb, v_emb)

        new_text_spans, indices = self.prune(sent_gen_text, attn_weights)

        for j in range(len(heterodata)):
          node_embs = sent_knowl_emb[j,indices[j],:]

          #get subj and relation of selected indices
          subj_ids = indices[j] // len(self.rels)
          pred_ids = indices[j] % len(self.rels)

          heterodata[j] = self.add_to_graph(heterodata[j], "csk", node_embs, sample_num=j, node_texts=new_text_spans[j])

          if i == 0:
            node_src_type = node_type
            node_tgt_type = "csk"
          else:
            node_src_type = node_tgt_type = "csk"

          for subj_id, pred_id, new_text_span in zip(subj_ids, pred_ids, new_text_spans[j]):
            subj = text_span[j][subj_id]
            pred = self.rels[pred_id]
            self.graph_as_dict[j][k].append((subj, pred, new_text_span))

          edge_type, edge_features, edge_index = self.create_edge_data(text_span, new_text_spans, (node_src_type, "edge", node_tgt_type), j)
          heterodata[j] = self.add_edges_to_graph(heterodata[j], edge_types, edge_features, edge_index)

          edge_type, edge_features, edge_index = self.create_edge_data(new_text_spans, [["total" for _ in subj_ids] for _ in new_text_spans], (node_tgt_type, "edge", "total"), j)
          heterodata[j] = self.add_edges_to_graph(heterodata[j], edge_types, edge_features, edge_index)

        text_spans[k] = new_text_spans

        gc.collect()

    if visualize:
      for i in range(len(heterodata)):
        fig = matplotlib.pyplot.figure()

        g = torch_geometric.utils.to_networkx(heterodata[i])
        networkx.draw(g, networkx.spring_layout(g, k=2), ax=fig.add_subplot())
        matplotlib.use("Agg") 
        fig.savefig(f"graph_png/{sent1[i]}_{i}.png")

    for g in heterodata:
      print("******")
      print(g["sent1"].x, g["sent1"].x.shape)
      print(g["sent2"].x, g["sent2"].x.shape)
      try:
        print(g["total"].x, g["total"].x.shape)
      except:
        print(g["total"])
        #print(a)
      print(g["csk"].x, g["csk"].x.shape)
    print("=====================")

    graph_batch = Batch.from_data_list(heterodata)
    x_dict = graph_batch.x_dict

    edge_index_dict = graph_batch.edge_index_dict

    for i, layer in enumerate(self.gnn_layers):
      x_dict = layer(x_dict, edge_index_dict) #, edge_attr=graph_batch.edge_attr_dict)
      if i == 0:
        x_dict = {key: torch.nn.functional.relu(value) for key, value in x_dict.items()}

    """k_1, v_1 = self.K1(node_embs1), self.V1(node_embs1)
    k_2, v_2 = self.K1(node_embs2), self.V1(node_embs2)
    q_1, q_2 = self.Q1(H_sent1), self.Q1(H_sent2)

    H_sent1, _ = self.multi_head_att(q_1, k_1, v_1) #, attn_mask=attn_mask1)
    H_sent2, _ = self.multi_head_att(q_2, k_2, v_2) #, attn_mask=attn_mask2)

    H_sent1 = H_sent1[:,0,:]
    H_sent2 = H_sent2[:,0,:]
    H_sent = torch.cat([H_sent1, H_sent2], dim=-1)"""

    #if visualize:
    #  return H_sent

    predictions = self.linear_layer(x_dict["total"])

    return predictions
