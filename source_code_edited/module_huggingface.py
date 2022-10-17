import torch
import torch.nn as nn
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompt_base import Template, Verbalizer
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from openprompt.utils import round_list, signature

from collections import namedtuple
#from model_center.model import BertConfig, RobertaConfig,T5, T5Config
#from model_center.tokenizer import T5Tokenizer, RobertaTokenizer
# from source_code_edited.roberta import Roberta
# from source_code_edited.bert import Bert
from transformers import RobertaForMaskedLM, RobertaConfig, RobertaTokenizer, BertForMaskedLM, BertConfig
from transformers import BertTokenizer,AutoTokenizer,XLMRobertaTokenizer,XLMTokenizer,BertweetTokenizer
#from source_code_edited.xlm_roberta_tokenizer import XLMTokenizer

#from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.plms.seq2seq import T5TokenizerWrapper
import bmtrain as bmt


ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))
_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper,}),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper}),
    'xlm': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper}),
    'bertweet': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': BertweetTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper}),
    'twitter-xlm': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper}),
    'chinese-roberta': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper}),
    'weibo-bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper}),
    }

def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]

def load_plm(model_name, model_path, specials_to_add = None,dtype=torch.float16):
    # print(model_name,model_path)
    # if model_path == "weibo-bert-base":
    #     model_path = '/home/zhujiangyi/.cache/model_center/weibo-bert-base'
    # if model_path == "weibo-bert-large":
    #     model_path = '/home/zhujiangyi/.cache/model_center/weibo-bert-large'
    # if model_path == "hfl/chinese-roberta-wwm-ext":
    #     model_path = '/home/zhujiangyi/.cache/model_center/chinese-roberta-wwm-ext'
    # if model_path == "hfl/chinese-roberta-wwm-ext-large":
    #     model_path = '/home/zhujiangyi/.cache/model_center/chinese-roberta-wwm-ext-large'
    model_class = get_model_class(plm_type = model_name)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    # if model_path =='xlm-roberta-base':
    #     model_path = '/home/zhujiangyi/.cache/model_center/xlm-roberta-base'
    # if model_path =='xlm-roberta-large':
    #     model_path = '/home/zhujiangyi/.cache/model_center/xlm-roberta-large'
    # if model_path == "vinai/bertweet-base":
    #     model_path = '/home/zhujiangyi/.cache/model_center/bertweet-base'
    # if model_path == "cardiffnlp/twitter-xlm-roberta-base":
    #     model_path = '/home/zhujiangyi/.cache/model_center/twitter-xlm-roberta-base'
    
    
    model_config = model_class.config.from_pretrained(model_path)
    if dtype == 'float32':
        model_config.half=False
        model_config.dtype=torch.float32
    model = model_class.model.from_pretrained(model_path, config=model_config)
    wrapper = model_class.wrapper

    return model, tokenizer, model_config, wrapper



class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``,
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``

    Args:
        plm (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        self.template = template
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

        self._prepare_main_input_name()

    def _prepare_main_input_name(self):
        model = self.plm
        if hasattr(model, "encoder") and hasattr(model.encoder, "main_input_name"):
            if model.encoder.main_input_name != model.main_input_name:
                main_input_name = model.encoder.main_input_name
            else:
                main_input_name = model.main_input_name
        else:
            main_input_name = getattr(model, "main_input_name", "input_ids")
        self.main_input_name = main_input_name

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        #################return_logits=True##################
        #outputs = self.plm(**input_batch, output_hidden_states=True, return_logits=True)
        outputs = self.plm(**input_batch)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures]) -> Dict:
        r"""Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.

    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the labels to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the label words (obtained by the current verbalizer).
        """
        outputs = self.prompt_model(batch)

        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits

    def predict(self):
        pass

    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    ##  We comment this code since it conflict with [OpenDelta](https://github.com/thunlp/OpenDelta)
    # def state_dict(self, *args, **kwargs):
    #     """ Save the model using template, plm and verbalizer's save methods."""
    #     _state_dict = {}
    #     if not self.prompt_model.freeze_plm:
    #         _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
    #     _state_dict['template'] = self.template.state_dict(*args, **kwargs)
    #     _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
    #     return _state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """ Load the model using template, plm and verbalizer's load methods."""
    #     if 'plm' in state_dict and not self.prompt_model.freeze_plm:
    #         self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
    #     self.template.load_state_dict(state_dict['template'], *args, **kwargs)
    #     self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")



