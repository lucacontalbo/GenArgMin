import torch
import spacy

from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from torch import nn

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
  
  def __init__(self, config, gen_steps=4):

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

    self.linear_layer = torch.nn.Linear(in_features=self.embed_size*2, out_features=self.num_classes)
    self.multi_head_att = torch.nn.MultiheadAttention(self.embed_size, 8, batch_first=True)
    self.Q = torch.nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
    self.K = torch.nn.Linear(in_features=1024, out_features=self.embed_size)
    self.V = torch.nn.Linear(in_features=1024, out_features=self.embed_size)

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

  def generate_knowledge(self, sents):
    sents_with_preds = []
    sample_boundaries = []
    for sent in sents:
      last_el = sample_boundaries[-1] if len(sample_boundaries) > 0 else 0
      sample_boundaries.append(last_el + len(sent)*len(self.rels))
      for span in sent:
        for rel in self.rels:
          sents_with_preds.append(span.format(rel))
    
    inputs = self.graph_tokenizer(sents_with_preds, return_tensors="pt", padding=True, truncation=True).to("cuda")
    print(inputs.shape)
    outputs = self.graph_model.generate(
        inputs["input_ids"],
        max_length=10,
        num_beams=5,
        num_return_sequences=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

    generated_text = self.graph_tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)

    return outputs, generated_text, sample_boundaries

  def get_knowledge_embedding(self, outputs, embed_type="all_tokens"):
    new_outputs = torch.tensor([], dtype=torch.float32).to("cuda") # to be changed on var
    if embed_type == "all_tokens":
      for hidden_states, sequence in zip(outputs["decoder_hidden_states"], torch.permute(outputs["sequences"][:,1:], (1,0))):
        # we mask padding tokens in generated sequence
        sequence = (sequence != 1).long().unsqueeze(-1).expand(len(sequence), 1024).unsqueeze(1)
        new_outputs = torch.cat([new_outputs, hidden_states[12]*sequence], dim=1)
        del sequence
    
    return new_outputs

  def group_batches(self, outputs, sample_boundaries):
    output = torch.zeros(self.batch_size, self.generated_max_length, 1024)
    start_index = 0

    assert len(sample_boundaries) == self.batch_size, f"sample_boundaries' length {len(sample_boundaries)} is not equal to the batch size {self.batch_size}"

    for i in range(len(sample_boundaries)):
      max_index = sample_boundaries[i]
      if max_index > 0:
        concatenated = outputs[start_index:start_index + max_index].reshape(-1, 1024)
        
        length = concatenated.shape[0]
        if length > self.generated_max_length:
          concatenated = concatenated[:self.generated_max_length]
        elif length < self.generated_max_length:
          padding = torch.zeros(self.generated_max_length - length, 1024)
          concatenated = torch.cat((concatenated, padding), dim=0)
            
        output[i] = concatenated
        start_index += max_index

    return output

  def prune(self, texts, attn_weights, sample_boundaries, seq_length):
    topk_sentences = []
    batched_sentences = []
    err = 0
    start_idx = 0

    for el in sample_boundaries:
      texts_in_sample = texts[start_idx:start_idx+el]
      batched_sentences.append(texts_in_sample)
      start_idx += el
    
    attn_weights = torch.mean(attn_weights, dim=1).squeeze(1)
    attn_weights_flat = attn_weights.view(attn_weights.shape[0], -1)

    # Get the top k values and their indices along the flattened dimension
    _, topk_indices = torch.topk(attn_weights_flat, self.k, dim=1)

    # Convert the flat indices back to 2D positions
    # Indices of shape (batch_size, k)
    row_indices = topk_indices // attn_weights.shape[-1]
    #col_indices = topk_indices % attn_weights.shape[-1]

    # Stack them to get a 3D tensor of shape (batch_size, k, 2)
    #topk_positions = torch.stack((row_indices, col_indices), dim=-1)
    row_indices = row_indices // seq_length

    for i in range(len(row_indices)):
      for j in range(len(row_indices[i])):
        try:
          topk_sentences.append(batched_sentences[i][row_indices[i][j]])
        except:
          err += 1
    
    print(f"Out of {self.batch_size * self.k} possible sentences, {err} have wrong indices")
    return topk_sentences


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

    sent1_spans = self.get_initial_text_spans(sent1)
    sent2_spans = self.get_initial_text_spans(sent2)

    for i in range(self.gen_steps):
      sent1_outputs, sent1_gen_text, sample_boundaries1 = self.generate_knowledge(sent1_spans)
      sent2_outputs, sent2_gen_text, sample_boundaries2 = self.generate_knowledge(sent2_spans)
      sent1_knowl_emb, sent2_knowl_emb = self.get_knowledge_embedding(sent1_outputs), \
                                         self.get_knowledge_embedding(sent2_outputs)
      sent1_knowl_emb = self.group_batches(sent1_knowl_emb, sample_boundaries1)
      sent2_knowl_emb = self.group_batches(sent2_knowl_emb, sample_boundaries2)

      k_1, v_1 = self.K(sent1_knowl_emb), self.V(sent1_knowl_emb)
      k_2, v_2 = self.K(sent2_knowl_emb), self.V(sent2_knowl_emb)
      q_1, q_2 = self.Q(H_sent1), self.Q(H_sent2)

      H_sent1, attn_weights1 = self.multi_head_att(q_1, k_1, v_1)
      H_sent2, attn_weights2 = self.multi_head_att(q_2, k_2, v_2)

      sent1_spans = self.prune(sent1_gen_text, attn_weights1, sample_boundaries1, len(sent1_outputs["sequences"][0]))
      sent2_spans = self.prune(sent2_gen_text, attn_weights2, sample_boundaries2, len(sent2_outputs["sequences"][0]))

    H_sent1 = H_sent1[:,0,:]
    H_sent2 = H_sent2[:,0,:]
    H_sent = torch.cat([H_sent1, H_sent2], dim=-1)

    #H_sent = torch.mean(att_output[0], dim=1)

    if visualize:
      return H_sent

    predictions = self.linear_layer(H_sent)

    return predictions