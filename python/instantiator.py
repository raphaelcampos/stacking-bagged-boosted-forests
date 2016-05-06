import numpy as np

from sklearn.base import clone

from config import base_estimators, default_params

class EstimatorInstantiator(object):
	""" """
	def __init__(self, base_estimators=base_estimators.copy(),
					 default_params=default_params.copy()):
		super(EstimatorInstantiator, self).__init__()
		
		self.base_estimators = base_estimators
		self.default_params = default_params

		if not self.base_estimators.keys() == self.default_params.keys():
			raise ValueError("base_estimators keys do not match default_parms ones")


	def get_instance(self, estimator, params=None):
		if not estimator in self.base_estimators:
			raise ValueError("unrecognized estimator: '%s'" % estimator)

		default_params = self.default_params[estimator].copy()

		if params is not None:
			default_params.update(params)


		return self.base_estimators[estimator]().set_params(
												**default_params)

	def get_estimators(self):
		return self.base_estimators.keys()

	def set_general_params(self, general_params):
		general_params_keys = set(general_params.keys())
		for estimator in default_params:
			params = set(base_estimators[estimator]().get_params().keys())
			keys = params & general_params_keys 
			for key in keys:
				self.default_params[estimator][key] = general_params[key]

	def set_params(self, params):
		for estimator in params:
			self.default_params[estimator].update(params[estimator])
