
from .stacking import StackingClassifier, VIG, DecisionTemplates, DirichletClassifier
from .boosting_forest import Broof, Bert
from .meta_outlier_removal import GaussianOutlierRemover, ThresholdOutlierRemover

__all__ = ['boosting_forest',
			'stacking'
			'VIG',
			'DecisionTemplates',
			'StackingClassifier',
			'Broof',
			'Bert']