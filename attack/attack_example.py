from abc import ABC, abstractmethod
import argparse

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from attack_Ci import AttackCiNN
from attack_Di import Conf_Attack, Pred_Attack


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CiとDi両方を使ってAiを攻撃")
    ap.add_argument("Ai_csv", help="Ai.csv")
    ap.add_argument("Ci_csv", help="Ci.csv")
    ap.add_argument("Di_json", help="Di.json(Booster.save_model)")
    ap.add_argument("-o", "--output", help="path_to_output", default="Fij.csv")
    args = ap.parse_args()

    attacker = MixAttack(args.Ci_csv, args.Di_json)
    attacker.infer(args.Ai_csv)
    attacker.save_inferred(args.output)
