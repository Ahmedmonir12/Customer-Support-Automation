import os,pathlib
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel,Field
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd 

load_dotenv()

CLIENT=OpenAI(base_url=os.getenv("LOCAL_URL"),api_key="llm")
MODEL_NAME="gemma4:e2b"
WORKERS=2

company_description="""ShopNest is a fictional online store selling electronics, home products, and fashion.

- Standard shipping: 3-5 business days, free for orders over $50.
- Express shipping: 1-2 business days, $12.99.
- Returns: Within 30 days, provided items are unused and in original packaging.
- Refunds: Processed within 5-7 business days.
- Payment: Visa, Mastercard, PayPal, and cash on delivery.
- Installments: Available for orders over $200.
- Support: 9 AM-9 PM daily; support@shopnest.example.
- Warranty: Electronics have a 12-month warranty; other products have a 6-month warranty.
- Loyalty: Silver, Gold, and Platinum tiers; customers earn 1 point per $1 spent.
"""

class CustomerRecord(BaseModel):
    orders:str=Field(
        description="you will write customer past orders, currently tracked order and changed or canceled orders."
        )
    product:str=Field(
        description="generate a product the customer ordered and give it a price and warranty"
    )
    support:str | None=Field(
        default=None,
        description="customer contact with support and his complaints for the product or service. this can be lift empty if the customer didn't contact support"
    )
    review:str=Field(
        description="the review the customer wrote for the order"
    )

class CustomersRecords(BaseModel):
    records:list[CustomerRecord]

def make_msg()->list[dict]:
    prompt=f"""
    You write help-center pages for a company.
    Here is the company facts {company_description}
    __
    Generate 10 different customers. Vary products, moods, and outcomes.
    About 3 to 6 customers should have contacted support.
    Customers who didn't contact support must have support set to null.
    Follow company policies in your generation.
    Fill the product field with a product and its catogary and price and warranty not just the catogary or a link.
    Ensure the order, product, support interaction, and review are consistent with each other.
    Support field must contain an actual conversation summary or case description not just an email address.
    Every customer must have an order description and don't use only an order ID or URL as the order description.
    """

    msg=[
        {
            "role":"system",
            "content":"You generate data for a fake company. Make the data vary from each other"
        },
        {"role":"user",
        "content":prompt
        }
        ]
    
    return msg

def generate_data(i):
    try:
            request=CLIENT.chat.completions.parse(
            messages=make_msg(),
            model=MODEL_NAME,
            response_format=CustomersRecords,
            temperature=0.6,
            reasoning_effort="low"
            )
            results=request.choices[0].message.parsed.records
            return [c.model_dump() for c in results]
    except Exception as e:
        print(f"Batch {i} failed: {e}")
        return
    


if __name__ == "__main__":
    with Pool(WORKERS) as pool:
        results=list(tqdm(pool.imap(generate_data, range(10)),total=10))
    rows = [row for batch in results for row in batch]
    pd.DataFrame(rows).to_csv("data/preprocessed/customers.csv", index=False)
