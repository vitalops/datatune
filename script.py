import os
import dask.dataframe as dd

import datatune.datatune as dt
from datatune.datatune.llm.llm import OpenAI
import time
from datatune.datatune.agent.agent import Agent

os.environ["OPENAI_API_KEY"] = "api_key"
#dask.config.set(scheduler='single-threaded')


llm = OpenAI(model_name="gpt-4o-mini-2024-07-18",tpm=1000000,rpm=5000)
df = dd.read_csv("products/sample.csv")
print(df.head())
prompt = (
    "You are a tabular data validation assistant.\n"
    "Given a 'contract' and few rows of data (including the header), generate python code to validate each row of data against the contract.\n"
    #"Some rules of the contract may depend on the previous rows of data to validate the current row. In such cases add required columns in a preprocessing step so that the validation can be done in a per row manner.\n"
    "Input data will be in the form of a python dataframe.\n"
    "Create new coloumn called valid and fill with yes if valid and no if invalid"
    "Here is the contract:\n"
 
    "Discounts\n"
    f"a) A flat 10% discount will be applied to all purchases that XYZ makes from ABC. This discount is calculated on the total Gross Amount of each transaction, regardless of the purchase volume.\n"

    f"Example: If XYZ places an order worth $50,000 (Gross Amount), a 10% discount would reduce the bill to $45,000.\n"

    f"b) If XYZ's total purchases (Gross Amount) from ABC in the previous calendar month exceed $80,000, an extra 1% discount will be applied on the Gross Amount in the current month. Additionally, for every additional $10,000 above the $80,000 threshold, XYZ will receive an extra 1% discount on Gross Amount, cumulatively.\n"

    "Example:\n"
    f"If XYZ spent $85,000 last month → 1% extra discount this month.\n"
    f"If they spent $100,000 → 3% extra discount this month (1% for exceeding $80,000, plus 2% for the additional $20,000).\n"
    f"If they spent $120,000 → 5% extra discount this month.\n"

    f"c) If the total Gross Amount of XYZ's purchases in the previous calendar quarter (Q-2) grew by 10% or more compared to the quarter before that (Q-1), then ABC will apply an additional 1% discount on Gross Amount on purchases made in the current quarter (Q-3).\n"

    "Example:\n"
    "Q2 Purchases = $90,000\n"
    f"Q3 Purchases = $99,000 → 10% growth\n"
    f"Since Q3 grew by exactly 10% over Q2, a 1% extra discount applies to Q4 purchases.)\n"
    )

agent = Agent(llm)
df = agent.do(prompt,df)


start = time.time()

# Transform data with Map
'''mapped = dt.Map(    
    prompt="Extract category and sub-category from industry",
    #prompt = "extract price condition (yes if more than 200)",
    output_fields=["Category","Sub-Category"],
    #output_fields=["Expensive"],
    input_fields=["Industry"]
   
)(llm, df)
# Filter data based on criteria
filtered = dt.Filter(
    prompt="keep only organizations that are in africa",
    #prompt = "keep only products that are in stock",
    input_fields=["Country"]

    
)(llm, mapped)'''

# Get the final dataframe after cleanup of metadata and deleted rows after operations using `finalize`.
result = dt.finalize(df)
result.compute().to_csv("electronics_products.csv")
end = time.time()

new_df = dd.read_csv("electronics_products.csv")
print(new_df.head())
print(f"⏱️ Total time taken: {end - start:.2f} seconds")

