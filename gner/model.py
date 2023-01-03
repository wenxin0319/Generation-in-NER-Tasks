import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForPreTraining
import ipdb
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        logger.info(f'Loading pre-trained model {config.model_name}')
        if config.model_name.startswith("copy+t5-"):
            model_name = config.model_name.split('copy+', 1)[1]
            self.model = T5Copy.from_pretrained(model_name, cache_dir=config.cache_dir, output_attentions=True)
            self.dec_start_id = 0
        elif config.model_name.startswith("copy+facebook/bart-"):
            model_name = config.model_name.split('copy+', 1)[1]
            self.model = BartCopy.from_pretrained(model_name, cache_dir=config.cache_dir, output_attentions=True)
            self.dec_start_id = 2
        elif config.model_name.startswith("facebook/bart-"):
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model = AutoModelForPreTraining.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            self.dec_start_id = 2
        elif config.model_name.startswith("t5-"):
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model = AutoModelForPreTraining.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            self.dec_start_id = 0
        else:
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model = AutoModelForPreTraining.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model_name = config.model_name
        self.prefix_fn_obj = None
        

    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs, 
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=4, max_length=50, desinated_prefix=""):
        self.eval()
        if desinated_prefix != "" and self.prefix_fn_obj is None:
            self.prefix_fn_obj = Prefix_fn_cls(desinated_prefix, self.tokenizer, self.dec_start_id)
        with torch.no_grad():
            if num_beams == 1:
                self.model._cache_input_ids = batch.enc_idxs
            else:
                expanded_return_idx = (
                    torch.arange(batch.enc_idxs.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(batch.enc_idxs.device)
                )
                input_ids = batch.enc_idxs.index_select(0, expanded_return_idx)
                self.model._cache_input_ids = input_ids
            
            if desinated_prefix != "":
                outputs = self.model.generate(input_ids=batch.enc_idxs, 
                                            attention_mask=batch.enc_attn, 
                                            num_beams=num_beams, 
                                            max_length=max_length,
                                            prefix_allowed_tokens_fn=lambda batch_id, sent: self.prefix_fn_obj.get(sent.tolist()))
            else:
                outputs = self.model.generate(input_ids=batch.enc_idxs, 
                                            attention_mask=batch.enc_attn, 
                                            num_beams=num_beams, 
                                            max_length=max_length)
            
        final_output = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()

        return final_output

class Prefix_fn_cls():
    def __init__(self, desinated_prefix, tokenizer, dec_start_id):
        self.tokenizer=tokenizer
        self.sub_tokens = [dec_start_id]+tokenizer(desinated_prefix, add_special_tokens=False)['input_ids']
        self.mapping = {}
        for idx in range(1, len(self.sub_tokens), 1):
            self.mapping[tuple(self.sub_tokens[:idx])] = self.sub_tokens[idx]
        self.full_set = list(range(len(tokenizer)))
    
    def get(self, previous_token):
        if len(previous_token) >= len(self.sub_tokens):
            return self.full_set
        previous_token = tuple(previous_token)
        if previous_token in self.mapping.keys():
            return self.mapping[previous_token]
        else:
            return self.full_set
        

class BartCopy(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.linear_copy = nn.Linear(config.d_model, 1)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == outputs['encoder_last_hidden_state'].size(0)
        except:
            ipdb.set_trace()

        # Copy distribution
        cross_attentions = outputs['cross_attentions'][-1] 
        cross_attentions = torch.mean(cross_attentions, dim=1) # batch x decoder_length x encoder_length

        # Probability of copying
        p_copy = torch.sigmoid(self.linear_copy(outputs['last_hidden_state']))
        
        # Merge distribution
        original_word_pro = torch.softmax(lm_logits, dim=-1) * (1 - p_copy) #[batch, sequence_length, vocab_size]
        copy_words = input_ids.unsqueeze(1).repeat(1, cross_attentions.size(1), 1) #(batch, target_length, encoder_length)
        lm_logits = torch.scatter_add(original_word_pro, 2, copy_words, cross_attentions*p_copy)
        
        eps = 1e-7
        lm_logits = torch.log(lm_logits+eps)

        masked_lm_loss = None
        if labels is not None:
            #loss_fct = CrossEntropyLoss()
            loss_fct = NLLLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class T5Copy(T5ForConditionalGeneration):
    
    def __init__(self, config):
        super().__init__(config)
        self.linear_copy = nn.Linear(self.model_dim, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == hidden_states.size(0)
        except:
            ipdb.set_trace()

        lm_logits = self.lm_head(sequence_output)
        
        # Copy distribution
        cross_attentions = decoder_outputs['cross_attentions'][-1] # batch x head x decoder_length x encoder_length
        cross_attentions = torch.mean(cross_attentions, dim=1) # batch x decoder_length x encoder_length

        # Probability of copying
        p_copy = torch.sigmoid(self.linear_copy(sequence_output))
        
        # Merge distribution
        original_word_pro = torch.softmax(lm_logits, dim=-1) * (1 - p_copy) #[batch, sequence_length, vocab_size]
        copy_words = input_ids.unsqueeze(1).repeat(1, cross_attentions.size(1), 1) #(batch, target_length, encoder_length)
        lm_logits = torch.scatter_add(original_word_pro, 2, copy_words, cross_attentions*p_copy)
        
        eps = 1e-7
        lm_logits = torch.log(lm_logits+eps)
        loss = None
        if labels is not None:
            #loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_fct = NLLLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )