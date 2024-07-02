import re
import pandas as pd
from sycamore.llms import OpenAIModels, OpenAI

def extract_year(question, company):
    pattern = r'\bFY\d{2}\b|\b\d{4}\b|\bFY\d{4}\b'
    yrs = (re.findall(pattern, question))

    yrs = [yr[-2:] for yr in yrs]

    year = ('20' + max(yrs)) if len(yrs) != 0 else ''

    return doc_exists(year, company)

def doc_exists(year, company):
    df = pd.read_csv('/home/admin/financebench_sample_150.csv')
    df = df['doc_name'].str.split('_', expand=True)
    
    df = df.loc[(df[0] == company) & (df[1].str.startswith(year))]

    return '' if df.empty else year

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
# prompt = RelevantInformationZeroShotGuidancePrompt()

# # with open("/home/admin/expert-knowledge.txt", 'r') as f:
# #     exp = f.read()

# def find_relevant_info(question, knowledge):#=exp):
#     rel_info = openai_llm.generate(
#         prompt_kwargs={
#             "prompt": prompt,
#             "question": question,
#             "knowledge": knowledge,
#         }
#     )

#     print (rel_info)
#     return (rel_info)

# find_relevant_info("Is 3M a capital-intensive business based on FY2022 data?", exp)

#########################################################################################

# from sycamore.llms import OpenAIModels, OpenAI
# from sycamore.llms.prompts import EntityExtractorFewShotGuidancePrompt
# import re
# import pandas as pd

# openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
# prompt = EntityExtractorFewShotGuidancePrompt()

# company_examples = (
#     "Entity Definition:\n"
#     "1. COMPANY: Short or long form name of an established company.\n"
#     "\n"
#     "Output Format:\n"
#     "{{'COMPANY': [list of entities present]}}\n"
#     "If no entities are presented in any categories keep it None\n"
#     "\n"
#     "Examples:\n"
#     "\n"
#     "1. Sentence: Is CVS Health a capital-intensive business based on FY2022 data?\n"
#     "Output: {{'COMPANY': ['CVS Health']}}\n"
#     "\n"
#     "2. Sentence: In 2022 Q2, which of JPM's business segments had the highest net income?\n"
#     "Output: {{'COMPANY': ['JP Morgan']}}\n"
#     "\n"
#     "3. Sentence: Has Microsoft increased its debt on balance sheet between FY2023 and the FY2022 period?\n"
#     "Output: {{'COMPANY': ['Microsoft']}}\n"
#     "\n"
#     "4. Sentence: Using the cash flow statement, answer the following question to the best of your abilities: how much did Block (formerly known as Square) generate in cash flow from operating activities in FY2020? Answer in USD millions.\n"
#     "Output: {{'COMPANY': ['Block']}}\n"
#     "\n"
#     "5. Sentence: {}\n"
#     "Output: "
# )

# year_examples = (
#     "Entity Definition:\n"
#     "1. YEAR: Any format of years without an associated month or date. Years can also be represented using only their last two digits. Years often follow 'FY'."
#     "\n"
#     "Output Format:\n"
#     "{{'YEAR': [list of entities present]}}\n"
#     "If no entities are presented in any categories keep it None\n"
#     "\n"
#     "Examples:\n"
#     "\n"
#     "1. Sentence: Is CVS Health a capital-intensive business based on FY2022 data?\n"
#     "Output: {{'YEAR': ['2022']}}\n"
#     "\n"
#     "2. Sentence: In 2022 Q2, which of JPM's business segments had the highest net income?\n"
#     "Output: {{'YEAR': ['2022']}}\n"
#     "\n"
#     "3. Sentence: Has Microsoft increased its debt on balance sheet between FY2023 and the FY2022 period?\n"
#     "Output: {{'YEAR': ['2023']}}\n"
#     "\n"
#     "4. Sentence: What drove revenue change as of the FY22 for AMD?\n"
#     "Output: {{'YEAR': ['2022']}}\n"
#     "\n"
#     "5. Sentence: From FY21 to FY22, excluding Embedded, in which AMD reporting segment did sales proportionally increase the most?\n"
#     "Output: {{'YEAR': ['2022']}}\n"
#     "\n"
#     "6. Sentence: Which business segment of JnJ will be treated as a discontinued operation from August 30, 2023 onward?\n"
#     "Output: {{'YEAR': ['2023']}}\n"
#     "\n"
#     "7. Sentence: What was the key agenda of the AMCOR's 8k filing dated 1st July 2022?\n"
#     "Output: {{'YEAR': ['2022']}}\n"
#     "\n"
#     "8. Sentence: {}\n"
#     "Output: "
# )

# def extract_company(final_prompt):
#     company = openai_llm.generate(
#         prompt_kwargs={
#                 "prompt": prompt,
#                 "entity": "company",
#                 "examples": company_examples,
#                 "query": final_prompt,
#             }
#     )

#     return company.replace(" ", "")

# def extract_year(final_prompt):
#     year = openai_llm.generate(
#         prompt_kwargs={
#                 "prompt": prompt,
#                 "entity": "year",
#                 "examples": year_examples,
#                 "query": final_prompt,
#             }
#     )
#     print (year)
#     return year

# def extract_year(question, company):
#     pattern = r'\bFY\d{2}\b|\b\d{4}\b|\bFY\d{4}\b'
#     yrs = (re.findall(pattern, question))

#     yrs = [yr[-2:] for yr in yrs]

#     year = ('20' + max(yrs)) if len(yrs) != 0 else ''
    
#     return doc_exists(year, company)

# def extract_both(final_prompt):
#     comp = extract_company(final_prompt)
#     yr = extract_year(final_prompt)

#     formatted = doc_exists(comp, yr)

#     return {
#             'company': formatted[0],
#             'year': formatted[1]
#             }

# def doc_exists(year, company):
#     df = pd.read_csv('/home/admin/financebench_sample_150.csv')
#     df = df['doc_name'].str.split('_', expand=True)

#     df = df.loc[(df[0] == company) & (df[1] == year)]
    
#     return '' if df.empty else year

# print (extract_year("Does AMEX have an improving operating margin profile as of 2022? If operating margin is not a useful metric for a company like this, then state that and explain why.", "AMERICANEXPRESS"))

# print (doc_exists('Activision Blizzard', '2019'))