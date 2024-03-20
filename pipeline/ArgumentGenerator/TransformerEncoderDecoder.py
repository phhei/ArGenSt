from abc import ABC
from typing import Optional, Tuple, Union

import transformers
import torch
from torch.nn import CrossEntropyLoss
from transformers import T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput

from loguru import logger


class T5ForConditionalArgGeneration(transformers.T5ForConditionalGeneration, ABC):
    def __init__(self, config: T5Config):
        super().__init__(config)

        logger.debug("OK, let's cast the encoder ({}) to the tailored class", self.encoder)
        tailored_encoder = self.encoder
        tailored_encoder.__class__ = T5EncoderForArg
        self.shift_insensitive_encoder = self.encoder
        self.encoder = tailored_encoder

    def get_encoder(self):
        logger.trace("We return the T5EncoderForArg instead of the original encoder: {}",
                     self.shift_insensitive_encoder)
        return super().get_encoder()

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, head_mask=None,
            decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None,
            encoder_outputs=None,
            encoder_output_shift: Optional[Union[Tuple[Tuple[torch.Tensor]], torch.FloatTensor]] = None,
            encoder_output_shift_impact: float = 1, **kwargs):
        ret_outputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, head_mask,
            decoder_head_mask, cross_attn_head_mask, use_cache,
            encoder_outputs, **kwargs
        )

        ret_outputs["encoder_output_shift"] = encoder_output_shift
        ret_outputs["encoder_output_shift_impact"] = encoder_output_shift_impact

        return ret_outputs

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None, decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
                encoder_output_shift: Optional[Union[Tuple[Tuple[torch.Tensor]], torch.FloatTensor]] = None,
                encoder_output_shift_impact: float = 1) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration
        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                encoder_output_shift=encoder_output_shift,
                encoder_output_shift_impact=encoder_output_shift_impact
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
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
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


class T5EncoderForArg(transformers.models.t5.modeling_t5.T5Stack, ABC):
    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                inputs_embeds=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                encoder_output_shift: Optional[Union[Tuple[Tuple[torch.Tensor]], torch.FloatTensor]] = None,
                encoder_output_shift_impact: float = 1):
        """
        A simple (encoder) forward pass

        :param input_ids: see huggingface-doc
        :param attention_mask: see huggingface-doc
        :param encoder_hidden_states: see huggingface-doc
        :param encoder_attention_mask: see huggingface-doc
        :param inputs_embeds: see huggingface-doc
        :param head_mask: see huggingface-doc
        :param cross_attn_head_mask: see huggingface-doc
        :param past_key_values: see huggingface-doc
        :param use_cache: see huggingface-doc
        :param output_attentions: see huggingface-doc
        :param output_hidden_states: see huggingface-doc
        :param return_dict: see huggingface-doc
        :param encoder_output_shift: [NEW] - how much should the final hidden states be adjusted?
        (incorporating knowledge from the author). Must be in shape of (#batch_size, #sequence_length, #embedding_size).
        The hidden state embedding size for t5-small is 512.
        :param encoder_output_shift_impact: [NEW] How much impact should the encoder_output_shift have.
        Default is 1 which means a simple addition of both original encoder hidden states and the shift. Here, you can
        scale the shift. BE AWARE: if the model is not used to incooperate such a shifting, the generation results will
        be odd. You should start fine-tuning with a decreased impact vector as .1
        :return:
        """
        encoder_outputs = super().forward(
            input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, inputs_embeds,
            head_mask, cross_attn_head_mask, past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict
        )

        if encoder_output_shift is None:
            logger.warning("You have a special instance of a T5-Model but you're not using it by missing tge input "
                           "\"encoder_output_shift\"")
        else:
            if isinstance(encoder_outputs, BaseModelOutput):
                last_hidden_state: torch.FloatTensor = encoder_outputs.last_hidden_state
            else:
                last_hidden_state: torch.FloatTensor = encoder_outputs[0]

            if isinstance(encoder_output_shift, Tuple):
                encoder_output_shift = torch.stack(
                    tensors=[torch.stack(s, dim=0) if isinstance(s, Tuple) else s for s in encoder_output_shift],
                    dim=0
                )
            try:
                last_hidden_state = last_hidden_state + encoder_output_shift_impact * encoder_output_shift
                if return_dict:
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=last_hidden_state,
                        hidden_states=encoder_outputs.hidden_states,
                        attentions=encoder_outputs.attentions,
                    )
                else:
                    encoder_outputs = (last_hidden_state,
                                       encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                                       encoder_outputs[2] if len(encoder_outputs) > 2 else None)
            except RuntimeError:
                logger.opt(exception=True).error("Failed to apply \"encoder_output_shift\": {}",
                                                 encoder_output_shift)

        return encoder_outputs
