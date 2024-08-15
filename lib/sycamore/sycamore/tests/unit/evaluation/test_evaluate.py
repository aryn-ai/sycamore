from sycamore.evaluation.evaluate import QualityAssessment
from sycamore.evaluation import EvaluationDataPoint


def test_run_evaluation():
    json_dict = {
        "data": [
            {
                "Question": "is this a Question ? ",
                "Answer": "This is an answer.",
                "SearchContexts": [
                    {
                        "document_url": "http://example.com",
                        "page_numbers": [1],
                        "document_id": "doc1",
                        "text_representation": "text",
                    }
                ],
                "Filters": {"filter1": "value1"},
            }
        ]
    }
    expected_result = {
        "doc": [
            {
                "properties": {},
                "elements": [],
                "raw": {
                    "Question": "is this a Question ? ",
                    "Answer": "This is an answer.",
                    "SearchContexts": [
                        {
                            "document_url": "http://example.com",
                            "page_numbers": [1],
                            "document_id": "doc1",
                            "text_representation": "text",
                        }
                    ],
                    "Filters": {"filter1": "value1"},
                },
                "ground_truth_answer": "This is an answer.",
                "filters": {"filter1": "value1"},
                "question": "None",
                "ground_truth_source_documents": [
                    {
                        "properties": {"_location": "http://example.com", "page_number": 1, "doc_id": "doc1"},
                        "text_representation": "text",
                    }
                ],
            }
        ]
    }

    result = QualityAssessment.create_evaluation_datapoint(json_dict)
    output = EvaluationDataPoint(result[0].get("doc"))

    assert output.get("raw").get("Question") == expected_result["doc"][0].get("raw").get("Question")
    assert output.get("raw").get("Answer") == expected_result["doc"][0].get("raw").get("Answer")
    assert output.get("raw").get("SearchContexts") == expected_result["doc"][0].get("raw").get("SearchContexts")
    assert output.get("ground_truth_answer") == expected_result["doc"][0].get("ground_truth_answer")
