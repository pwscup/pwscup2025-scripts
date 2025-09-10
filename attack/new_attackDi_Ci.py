from abc import ABC, abstractmethod
import argparse

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from attack_Ci import AttackCiNN
from attack_Di import Conf_Attack, Pred_Attack

"""
    CiとDi両方を使ってAiを攻撃
    具体的には、Diを使ってAiの出力を推論し、その結果を基にCiを使って順位付けを行う。
    new_attackDi_Ci: score num_n Di_attack answers by Ci_attack distance
    new_attackDi_Ci: output: 1st-ranked record
"""

class AttackBase(ABC):
    def __init__(self, path_to_Ci_csv, path_to_Di_json):
        self.inferred = None
    
    @abstractmethod
    def infer(self, path_to_Ai_csv):
        pass

    def save_inferred(self, path_to_output):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path_to_output, index=False, header=False)
            print(f"inferred was successfully saved as {path_to_output}")

class MixAttack(AttackBase):
    def __init__(self, path_to_Ci_csv, path_to_Di_json):
        super().__init__(path_to_Ci_csv, path_to_Di_json)
        self.attacker1 = AttackCiNN(path_to_Ci_csv)
        self.attacker2 = Conf_Attack(path_to_Di_json)
        self.attacker3 = Pred_Attack(path_to_Di_json)

    def infer(self, path_to_Ai_csv):
        # 3つの攻撃器でそれぞれ攻撃
        inferred1 = self.attacker1.infer(path_to_Ai_csv)
        inferred2 = self.attacker2.infer(path_to_Ai_csv)
        inferred3 = self.attacker3.infer(path_to_Ai_csv)
        
        # 3つの攻撃結果で多数決
        inferred = pd.DataFrame(inferred1 + inferred2 + inferred3 >1.5, dtype=int)
        self.inferred = inferred

        return inferred