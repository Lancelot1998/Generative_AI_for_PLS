# test_bruteforce.py
from env_legit_link import LegitLinkGenerator
from qwen_eve_utils import build_eve_models_from_qwen
from search_bruteforce import brute_force_power_and_k

model_path = "/models/Qwen"   # 你下载的模型路径

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )

k = [4,8,16,32,64]

legit_gen = LegitLinkGenerator(k)

k_to_params = legit_gen.k_to_params

bob_models, alice_pos, bob_pos, d_bob = legit_gen.build_bob_semantic_models(k_to_params)

result = gen_Qwen(model,tokenizer)

qwen_eve_info = json.loads(result)

eve_models, eve_pos, d_eve = build_eve_models_from_qwen(qwen_eve_info, k_to_params)

# 4. 定义 Alice 的可选发射功率
P_list = [0.1 * i for i in range(1, 11)]  # 0.1 ~ 1.0

# 5. 穷举搜索最佳 (P,k)
best_P, best_k, xi_B, xi_E, R = brute_force_power_and_k(
    bob_models, eve_models,
    P_list=P_list,
    distance_bob=d_bob,
    distance_eve=d_eve,
    sigma2_bob=0.1,
    sigma2_eve=0.1,
    n_mc=200,
    lambda_e=1.0,
    bob_xi_min=0.5  # 要求 Bob 至少 0.5
)

print("Alice 位置:", alice_pos)
print("Bob   位置:", bob_pos, " 距离:", d_bob)
print("Eve   位置:", eve_pos,  " 距离:", d_eve)
print("Best P:", best_P)
print("Best k:", best_k)
print("Bob xi:", xi_B)
print("Eve xi:", xi_E)
print("Reward (SSR):", R)
