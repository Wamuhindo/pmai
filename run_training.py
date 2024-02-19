import os

scenario_names = [
                #"Apps_400_3",
                #"Apps_500_3",
                #"Apps_600_3",
                #"Apps_400_3_not",
                #"Apps_500_3_not",
                #"Apps_600_3_not" 
                #"Apps",
                #"Apps_1_3T",
                #"Apps_2_3T",
                #"Apps_3_3T",
                #"Apps_400_1_not_unique",
                #"Apps_400_2_not_unique",
                #"Apps_400_3_not_unique",
                #"Apps_500_1_not_unique",
                #"Apps_500_2_not_unique",
                #"Apps_500_3_not_unique",
                #"Apps_600_1_not_unique",
                #"Apps_600_2_not_unique",
                #"Apps_600_3_not_unique",
                #"Apps_400_1_unique",
                #"Apps_400_2_unique",
                #"Apps_400_3_unique",
                #"Apps_1_3T_not",
                #"Apps_2_3T_not",
                #"Apps_3_3T_not",
                "Apps_All_Q_400",
                "Apps_All_Q_500",
                "Apps_All_Q_600",
                "Apps_All_Q_3T_W1",
                "Apps_All_Q_3T_W2",
                "Apps_All_Q_3T_W3",
             ]

for scenario in scenario_names:
    print(f"running directory {scenario} ...")
    cmd = f"python3 main_checkpoint2.py -l logs_train_{scenario}.txt -A Apps_all/{scenario}"
    os.system(cmd)