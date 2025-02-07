import pandas as pd
from src.company_matching import CompanyMatcher

# Load Data
G = pd.read_csv("data/G.csv", delimiter="|", encoding="utf-8")
STrain = pd.read_csv("data/STrain.csv", delimiter="|", encoding="utf-8")
STest = pd.read_csv("data/STest.csv", delimiter="|", encoding="utf-8")

# Initialize Matcher
matcher = CompanyMatcher(G)

# Train Model on STrain
train_predictions = matcher.match_companies(STrain)
train_predictions.to_csv("output/STrain_final_predictions.csv", sep="|", index=False)

# Run Matching on STest (Matches `sample_submission.csv` format)
test_predictions = matcher.match_companies(STest, test_mode=True)
test_predictions.to_csv("output/STest_final_predictions.csv", sep="|", index=False)

print("✅ Matching Completed. Results saved in 'output/' folder.")