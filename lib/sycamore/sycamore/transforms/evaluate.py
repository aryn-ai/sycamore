import logging
from typing import Any, Callable, Iterable, Optional, Dict

from abc import ABC, abstractmethod

import os  
import sycamore
from sycamore.reader import DocSetReader
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.evaluation import EvaluationDataPoint, EvaluationMetric
from sycamore.evaluation.metrics import document_retrieval_metrics, rouge_metrics

class Assessment(ABC):
	@abstractmethod
	def run_evaluation(self, **kwargs):
		pass
	
	def run(self, index):
		return self.run_evaluation(index)	

	@staticmethod
	def create_evaluation_datapoint(json_dict: dict[str, Any]):
		## find out why removing static upsets it ? 
		result = []
		print('create evaluation datapoint')
		logging.error(json_dict)
		assert json_dict is not None
		assert isinstance(json_dict,dict)
		for datapoint in json_dict["data"]:
			document = EvaluationDataPoint()
			document.raw = datapoint
			document.question = datapoint["Question"]    
			document.ground_truth_answer = datapoint["Answer"]
			document.filters = datapoint.get("Filters", None)
			for filter in document.filters.keys():
				document.filters[filter] = datapoint.get("Filters").get(filter)
			
			result += [{"doc": document.serialize()}]
		return result
	
class QualityAssessment(Assessment): 
	def __init__(self, GT_path: str , rag_config: Dict , **kwargs ) :
		self.user = os.environ.get('USER', os.environ.get('USERNAME'))
		self.GT_path = GT_path
		self.rag_config = rag_config
		self.os_client_args = kwargs.get("os_client_args", "") 
		self.ctx = sycamore.init()
		if kwargs.get("metrics", "") == "":
			self.metrics = [document_retrieval_metrics, rouge_metrics]
		else:
			self.metrics = kwargs.get("metrics", "")


	def run_evaluation(self,index, *args, **kwargs):
		print('in run evaluation',self.GT_path)
		# print(self.create_evaluation_datapoint(self.GT_path))
		input_docset = DocSetReader(self.ctx).json(paths =self.GT_path, doc_extractor = QualityAssessment.create_evaluation_datapoint)
		input_docset.show()
		pipeline = EvaluationPipeline(index = index
									, os_client_args = self.os_client_args
									, os_config = self.rag_config
									, metrics = self.metrics )
		query_level_metrics, aggregated_metrics = pipeline.execute(input_docset)

		return query_level_metrics, aggregated_metrics
		

	
class Evaluate():
	def __init__(self, index: str, assessment: Optional[Assessment], **kwargs):
		super().__init__()
		assessment.run(index)
	 
		 