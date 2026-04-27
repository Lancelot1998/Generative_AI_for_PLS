from test_drl import z_main
from test_drl_nollm import main_random_multi
from test_Qwen_teacher import main_qwen_teacher

if __name__ == "__main__":
    main_random_multi()
    z_main()
    main_qwen_teacher()