"""Validation-only selection. Never splice metrics from different checkpoints."""
import math


class CheckpointSelection:
    def __init__(self, target_auc=0.871, target_mae=0.089):
        self.target_auc,self.target_mae=target_auc,target_mae
        self.best_mae=float("inf")
        self.best_auc=-float("inf")
        self.best_joint=float("inf")
        self.best_feasible_auc=-float("inf")

    def update(self, metrics):
        auc,mae=metrics["auc"],metrics["mae"]
        if not math.isfinite(mae) or (auc is not None and not math.isfinite(auc)):
            raise ValueError("Nonfinite selection metrics")
        joint=(max(0,self.target_auc-auc)/self.target_auc +
               max(0,mae-self.target_mae)/self.target_mae) if auc is not None else float("inf")
        improved=dict(mae=mae < self.best_mae, auc=auc is not None and auc > self.best_auc,
                      joint=joint < self.best_joint,
                      feasible_auc=auc is not None and mae < self.target_mae and auc > self.best_feasible_auc)
        if improved["mae"]: self.best_mae=mae
        if improved["auc"]: self.best_auc=auc
        if improved["joint"]: self.best_joint=joint
        if improved["feasible_auc"]: self.best_feasible_auc=auc
        return improved

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self,state):
        self.__dict__.update(state)
