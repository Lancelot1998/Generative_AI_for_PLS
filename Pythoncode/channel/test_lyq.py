from env_legit_link import LegitLinkGenerator
from qwen_eve_utils import build_eve_models_from_qwen
from search_bruteforce import brute_force_power_and_k

k = [4,8,16,32,64]
legit_gen = LegitLinkGenerator(k)
print(legit_gen.k_to_params[4])
print(gen_Qwen)