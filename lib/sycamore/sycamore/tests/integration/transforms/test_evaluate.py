
import pytest
from sycamore.transforms.evaluate import QualityAssessment,Evaluate


class TestTransformEvaluate:
    INDEX = ""

    OS_CLIENT_ARGS = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": False,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    OS_CONFIG = {
        "size": 10,
        "neural_search_k": 100,
        "embedding_model_id": "SE1lDZABqmytCSGjsh1L",
        "search_pipeline": "hybrid_rag_pipeline",
        "llm": "gpt-4-turbo",
        "context_window": "5",
    }
    @pytest.mark.skip(reason="Requires named models to configure os pipeline unless we setup the cluster on each run")
    def test_pipeline(self):
        custom_question_augmentation = "{}, The product code is {}."
        question_augmentation_filter = 'properties._product_codes'
        assessment = QualityAssessment(
            os_client_args=self.OS_CLIENT_ARGS, 
            rag_config= self.OS_CONFIG, GT_path = './part_lookups.json', 
            custom_question_augmentation=custom_question_augmentation, 
            question_augmentation_filter = question_augmentation_filter)
        evaluate = Evaluate('5_sram_syca_openai_star_product_codes_20th',assessment)
        print(evaluate.result)