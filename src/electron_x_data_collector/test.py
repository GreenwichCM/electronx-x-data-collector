import liberator
import os
from dotenv import load_dotenv

load_dotenv()

liberator.url = os.getenv("API_BASE_URL")
username      = os.getenv("USERNAME")
api_key       = os.getenv("API_TOKEN")

query = liberator.query(
    name = "",
    as_of = "",
    back_to = "",
    user = username,
    token = api_key
)

if __name__ == "__main__":
    df = liberator.get_dataframe(query)