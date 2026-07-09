# this whole file exists because balcony-llama returns logits as a tuple
# (one per exit + full model) but adversariallm expects a single tensor.
# i solve this by rebinding the model's __class__ in place so i don't
# have to reload weights or copy anything.

from transformers.modeling_outputs import CausalLMOutputWithPast

from .balcony_llama.modeling_nestedllama import NestedLlamaForCausalLM


class SingleExitCausalLM(NestedLlamaForCausalLM):
    """wraps the balcony checkpoint so only one exit's logits are exposed at a time.
    call set_active_exit() to switch which exit forward() returns."""

    _active_exit = "full"

    def available_exits(self):
        return list(self.config.exit_layer_indices) + ["full"]

    def set_active_exit(self, exit_layer):
        if exit_layer not in self.available_exits():
            raise ValueError(f"{exit_layer!r} not in {self.available_exits()}")
        self._active_exit = exit_layer

    def _exit_index(self):
        # the logits tuple is ordered: [exit_layer_0, exit_layer_1, ..., full_model]
        # so "full" is always the last element
        if self._active_exit == "full":
            return len(self.config.output_exit_layers)
        return self.config.output_exit_layers.index(self._active_exit)

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        outputs = super().forward(*args, **kwargs)
        selected_logits = outputs.logits[self._exit_index()]
        return CausalLMOutputWithPast(
            loss=None,
            logits=selected_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def wrap_for_single_exit(model: NestedLlamaForCausalLM) -> SingleExitCausalLM:
    """rebind an already-loaded model's class in place (no reload, no copy)."""
    # i need all exits computed every forward pass so i can switch freely
    if model.config.output_exit_layers != list(model.config.exit_layer_indices):
        raise ValueError(
            "expected config.output_exit_layers to request every exit "
            f"(got {model.config.output_exit_layers}, "
            f"exit_layer_indices={model.config.exit_layer_indices})"
        )
    if not model.config.output_full_model:
        raise ValueError("expected config.output_full_model=True")
    model.__class__ = SingleExitCausalLM
    model.set_active_exit("full")
    return model


class PlainCausalLM(NestedLlamaForCausalLM):
    """for the no-balcony checkpoint (parsakaveh/LLM-1B-Pretrained).
    it has exit_layer_indices=[] so the logits tuple has exactly one element,
    but its still a tuple not a tensor. this unwraps it so adversariallm
    doesn't choke. i only found this bug by actually trying to attack the
    plain model, reading the code alone wouldn't have caught it."""

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        outputs = super().forward(*args, **kwargs)
        assert len(outputs.logits) == 1, (
            f"expected exactly one (full-model) logits tensor, got {len(outputs.logits)} "
            "-- this checkpoint may have exit layers configured after all"
        )
        return CausalLMOutputWithPast(
            loss=None,
            logits=outputs.logits[0],
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def wrap_plain_model(model: NestedLlamaForCausalLM) -> PlainCausalLM:
    """rebind an already-loaded no-exit checkpoint's class in place."""
    if model.config.exit_layer_indices:
        raise ValueError(f"expected no exit layers, got {model.config.exit_layer_indices!r}")
    if model.config.output_exit_layers is not None:
        raise ValueError(f"expected output_exit_layers=None, got {model.config.output_exit_layers!r}")
    if not model.config.output_full_model:
        raise ValueError("expected config.output_full_model=True")
    model.__class__ = PlainCausalLM
    return model
