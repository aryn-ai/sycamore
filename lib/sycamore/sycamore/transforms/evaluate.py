import logging
from typing import Any, Callable, Iterable, Optional, Dict, List, Union

from abc import ABC, abstractmethod

import os  
import sycamore
from sycamore.reader import DocSetReader
from sycamore.data import Document
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.evaluation import EvaluationDataPoint, EvaluationMetric
from sycamore.evaluation.metrics import document_retrieval_metrics, rouge_metrics

class Assessment(ABC):
	@abstractmethod
	def run_evaluation(self, **kwargs):
		pass
	
	def __call__(self, index):
		return self.run_evaluation(index)	
	
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
		self.custom_question_augmentation = kwargs.get("custom_question_augmentation",{})
		self.question_augmentation_filter = kwargs.get("question_augmentation_filter","")


	@staticmethod
	def create_evaluation_datapoint( json_dict: dict[str, Any], custom_question_augmentation: Dict  = {}, question_augmentation_filter: str  = ""  ):
		result = []
		assert json_dict is not None
		assert isinstance(json_dict,dict)
		for datapoint in json_dict["data"][:]:
			document = EvaluationDataPoint()
			document.raw = datapoint
			document.ground_truth_answer = datapoint["Answer"]
			document.filters = datapoint.get("Filters", None)
			for filter in document.filters.keys():
				document.filters[filter] = datapoint.get("Filters").get(filter)
			document.question = custom_question_augmentation.format(document.question,document.filters.get(question_augmentation_filter)) 
			source_documents = []
			for search_result in datapoint["SearchContexts"]:
				source_document = Document()
				properties = {
					"_location": search_result["document_url"],
					"page_number": search_result["page_numbers"][0]
				}
				source_document.properties = properties
				source_document.text_representation = search_result["text_representation"]
				source_document.doc_id = search_result["document_id"]
				source_documents += [source_document]
			document.ground_truth_source_documents = source_documents

			result += [{"doc": document.serialize()}]
		return result

	def run_evaluation(self,index: str, *args, **kwargs):
		custom_question_augmentation = str(self.custom_question_augmentation)
		question_augmentation_filter = str(self.question_augmentation_filter)
		input_docset = DocSetReader(self.ctx).json(paths =self.GT_path, doc_extractor= lambda json_dict: QualityAssessment.create_evaluation_datapoint(json_dict, custom_question_augmentation, question_augmentation_filter))
		pipeline = EvaluationPipeline(index = index
									, os_client_args = self.os_client_args
									, os_config = self.rag_config
									, metrics = self.metrics )
		query_level_metrics, aggregated_metrics = pipeline.execute(input_docset)

		return query_level_metrics.take_all(), aggregated_metrics

class Evaluate():
	"""
	The Evaluate Transform runs the evaluation test for Question Answering on 
	Index or list of indices against a ground truth
	"""
	def __init__(self, index: Union[str, List[str]], assessment: Optional[Assessment], **kwargs):
		print(type(index))
		super().__init__()
		if isinstance(index, List) and all(isinstance(i, str) for i in index):
			self.result =  {sublist: assessment(index)for idx in index}
		elif isinstance(index, str):
			self.result =  {index: assessment(index)}
		else:
			raise ValueError("Input must be a str or a list of str")
