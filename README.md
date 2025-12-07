Task 1: 訓練 Extended HRV (進階生理訊號)
對應原本的 test2_data2_binary_hrv_Multimorbidity.py

Bash

python main.py --task train --mode extended_hrv
Task 2: 訓練 Basic HRV (基礎生理訊號 + SHAP)
對應原本的 test2_data2_binary_Multimorbidity.py

Bash

python main.py --task train --mode basic_hrv
Task 3: 訓練 Psych (僅心理量表)
對應原本的 test2_data2_binary_psych_Multimorbidity.py

Bash

python main.py --task train --mode psych
Task 4: 訓練 All Features (全特徵融合)
對應原本的 test2_data2_binary_all_Multimorbidity.py

Bash

python main.py --task train --mode all
Task 5: 執行外部驗證
對應原本的 external_validate_A_Data1_Multimorbidity.py 注意：這裡的 Run_xxx 資料夾名稱必須替換成您實際訓練跑出來的資料夾名稱。

Bash

python main.py --task validate --model_dir "Run_basic_hrv_20251129_210509"