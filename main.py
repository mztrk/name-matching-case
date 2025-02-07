import pandas as pd
from src.company_matching import CompanyMatcher
from src.utils import process_matching

# Load Data
G = pd.read_csv("data/G.csv", delimiter="|", encoding="utf-8")

# Initialize Matcher
matcher = CompanyMatcher(G)

# Process STrain (Training Data with company_id)
process_matching(matcher, "data/STrain.csv", "output/matched_results_train.csv")

# Process STest (Test Data without company_id)
process_matching(matcher, "data/STest.csv", "output/matched_results_test.csv")