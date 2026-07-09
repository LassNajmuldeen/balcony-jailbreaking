# i register the model type locally so AutoModelForCausalLM.from_pretrained()
# picks up my vendored copy instead of downloading code from the hub.
# this way i run code i've actually read, not trust_remote_code=True.

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_nestedllama import NestedLlamaConfig
from .modeling_nestedllama import NestedLlamaForCausalLM

AutoConfig.register("nested_llama", NestedLlamaConfig)
AutoModelForCausalLM.register(NestedLlamaConfig, NestedLlamaForCausalLM)
