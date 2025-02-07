import os
import time
import pandas as pd

def standardize_company_name(name):
    """Normalize company names while selectively keeping legal structures."""
    if pd.isnull(name): return ""
    name = name.lower().strip()
    legal_terms = ["llc", "inc", "ltd", "s.p.a.", "s.r.l."]
    for term in legal_terms:
        if name.endswith(term):
            continue  
        name = name.replace(term, "").strip()
    return name

def calculate_cost(row):
    """
    Compute the cost function per row.
    Args:
        row (pd.Series): A single row containing 'company_id' (true) and 'company_id_pred' (predicted).
    
    Returns:
        cost (int): Cost for this row based on the given function.
    """
    true_id = row["company_id"]
    pred_id = row["company_id_pred"]
    
    if true_id == pred_id:
        return 0  # Correct match (Best Case)
    elif pred_id == -1:
        return 1  # Incorrectly classified as "not in G"
    else:
        return 5  # Incorrectly matched to another company (Worst Case)

def process_matching(matcher, input_file, output_file):
    """
    Matches companies in the given dataset against G and saves results, tracking execution time.
    
    Args:
        matcher (CompanyMatcher): Instance of the matching class.
        input_file (str): Path to input CSV file (STrain or STest).
        output_file (str): Path to save the output CSV file.
    """
    print(f"Processing {input_file} ...")
    start_time = time.time()  # Start timing

    # Load dataset
    S = pd.read_csv(input_file, delimiter="|", encoding="utf-8")

    # Check if company_id column exists (for cost calculation)
    if "company_id" in S.columns:
        print("company_id found - computing cost after matching.")
        matched_results = matcher.match_companies(S, test_mode=False)  # Keep company_id
        matched_results["cost_function"] = matched_results.apply(calculate_cost, axis=1)

        # Print Cost Function Summary
        cost_summary = matched_results["cost_function"].value_counts().to_dict()
        print("Cost Function Occurrences:", cost_summary)
    else:
        print("No company_id found - running in test mode.")
        matched_results = matcher.match_companies(S, test_mode=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save Results
    matched_results.to_csv(output_file, sep="|", index=False)
    
    # End timing
    end_time = time.time()
    execution_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"Execution Time for {input_file}: {execution_time:.2f} minutes\n")