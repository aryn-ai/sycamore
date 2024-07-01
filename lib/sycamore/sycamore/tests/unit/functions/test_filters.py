from sycamore.functions.filters import llm_filter, match_filter, range_filter
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.data import Document


class TestFilters:
    def test_llm_filter(self):
        client = OpenAI(OpenAIModels.GPT_4O.value)
        doc = Document()
        filter_question = "Was this incident caused due to environmental factors?"
        env_text = """The pilot reported that she encountered unexpected winds while 
        flying at night. After multiple encounters with strong wind gusts and turbulence,
         the pilot lost control of the airplane and it started to spin downward."""
        non_env_text = """The student pilot reported that, during landing, the airplane bounced. 
        She added power to go around, pitched the yoke of the airplane back too aggressively, and 
        the airplane stalled."""
        assert llm_filter(client, doc, filter_question, env_text)
        assert llm_filter(client, doc, filter_question, non_env_text) is False

    def test_match_filter(self):
        query = 3
        nums = [1, 3, 5, 9, 3, 2, 4]
        for i in range(len(nums)):
            if i == 1 or i == 4:
                assert match_filter(query, nums[i])
            else:
                assert match_filter(query, nums[i]) is False

        query = "sub"
        texts = ["submarine", None, "awesome", True, "unsubtle", "sub", "sunny", "", 4]
        filtered_texts = []
        for t in texts:
            if match_filter(query, t):
                filtered_texts.append(t)
        assert filtered_texts == ["submarine", "unsubtle", "sub"]

    def test_range_filter(self):
        start, end = 2, 4
        nums = [1, 3, 5, 9, 3, 2, 4]
        filtered_nums = []
        for n in nums:
            if range_filter(start, end, n):
                filtered_nums.append(n)
        assert filtered_nums == [3, 3, 2, 4]
