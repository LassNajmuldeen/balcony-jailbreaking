# this whole file exists because balcony-llama hands back logits as a tuple (one
# per exit, plus the full model) while adversariallm wants a single tensor. i fix
# that by rebinding the model's __class__ in place, so theres no reloading of
# weights and nothing gets copied.
#
# the upshot is that every attack, eval and judge script downstream can treat one
# exit as if it were just an ordinary causal LM.

from transformers.modeling_outputs import CausalLMOutputWithPast

from .balcony_llama.modeling_nestedllama import NestedLlamaForCausalLM


class SingleExitCausalLM(NestedLlamaForCausalLM):
    """exposes one exit's logits at a time out of a balcony checkpoint.
    call set_active_exit() to change which exit forward() returns."""

    # class attribute, but set_active_exit writes it on the instance. the default
    # matters because a freshly wrapped model has to be usable before anyone calls
    # set_active_exit
    _active_exit = "full"

    def available_exits(self):
        return list(self.config.exit_layer_indices) + ["full"]

    def set_active_exit(self, exit_layer):
        # validated here rather than at forward time, so a typo shows up straight
        # away instead of half way through an attack
        if exit_layer not in self.available_exits():
            raise ValueError(f"{exit_layer!r} not in {self.available_exits()}")
        self._active_exit = exit_layer

    def _exit_index(self):
        # the logits tuple is ordered [exit_layer_0, exit_layer_1, ..., full_model]
        # so "full" is always the last element
        if self._active_exit == "full":
            return len(self.config.output_exit_layers)
        return self.config.output_exit_layers.index(self._active_exit)

    def forward(self, *args, **kwargs):
        # forced, otherwise the parent can return a plain tuple and .logits below
        # would blow up
        kwargs["return_dict"] = True
        outputs = super().forward(*args, **kwargs)
        selected_logits = outputs.logits[self._exit_index()]
        # loss=None on purpose. the callers compute their own loss over their own
        # slices, and a loss computed here would be over the wrong positions
        return CausalLMOutputWithPast(
            loss=None,
            logits=selected_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def wrap_for_single_exit(model: NestedLlamaForCausalLM) -> SingleExitCausalLM:
    """rebinds an already loaded model's class in place. no reload, no copy."""
    # every exit has to be computed on every forward pass, otherwise switching
    # exits between calls would silently return whichever ones the config asked
    # for. checked rather than assumed, because a checkpoint can ship with a
    # narrower output_exit_layers than it has exits
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

    it has exit_layer_indices=[] so the logits tuple holds exactly one element,
    but its still a tuple and not a tensor, and this unwraps it so adversariallm
    doesnt choke on it. i only found that by actually trying to attack the plain
    model, reading the code alone would not have caught it.
    """

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        outputs = super().forward(*args, **kwargs)
        # an assert and not a silent [0]: more than one element would mean this
        # checkpoint does have exits after all, and the "baseline" would quietly
        # be measuring one of them
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
    """rebinds an already loaded no-exit checkpoint's class in place.

    only the 1B baseline (parsakaveh/LLM-1B-Pretrained) is a nested checkpoint
    that needs this. the 7B/8B baselines are stock HF llama and their forward
    already returns a logits tensor, so rebinding those to PlainCausalLM would
    make the len()==1 assert read the BATCH dimension and then silently drop it.
    so anything that isnt a NestedLlama is handed straight back untouched.
    """
    if not isinstance(model, NestedLlamaForCausalLM):
        return model
    # these three checks are what "this really is the no-exit checkpoint" means.
    # getting the wrong model here would invalidate every baseline vs exit number
    if model.config.exit_layer_indices:
        raise ValueError(f"expected no exit layers, got {model.config.exit_layer_indices!r}")
    if model.config.output_exit_layers is not None:
        raise ValueError(f"expected output_exit_layers=None, got {model.config.output_exit_layers!r}")
    if not model.config.output_full_model:
        raise ValueError("expected config.output_full_model=True")
    model.__class__ = PlainCausalLM
    return model
