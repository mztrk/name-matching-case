# Company Name Matching

This project matches external company names with a ground-truth database, using **TF-IDF vectorization, Nearest Neighbors, and Fuzzy Matching**.

## Features
- **Handles new data easily**
- **TF-IDF + Nearest Neighbors for initial matching**
- **Fuzzy Matching to fix typos**
- **Generates output matching `sample_submission.csv` format**

---

## Project Structure
```
company_matching_project/
│── main.py                   # Main script to run training and testing
│── requirements.txt           # Dependencies for Python
│── Dockerfile                 # Docker setup (Optional)
│── README.md                  # Documentation
│
├── FuzzyMatch.ipynb           # Matching model implementation     
│
├── src/                       # Source code
│   ├── company_matching.py    # Matching model implementation
│   ├── utils.py               # Helper functions (preprocessing, phonetics, etc.)
│
├── data/                      # Data files
│   ├── G.csv                  # Ground truth dataset
│   ├── STrain.csv             # Training dataset
│   ├── STest.csv              # Test dataset (for new predictions)
│   ├── sample_submission.csv  # Expected test submission format
│
├── output/                    # Output folder
│   ├── matched_results_train.csv   # Training results
│   ├── matched_results_test.csv    # Test results (matches `sample_submission.csv` format)
---
```

## How to Run
### ** Option 1: Run Locally with Python & Virtual Environment (`venv`)**

#### **1. Create a Virtual Environment**

Navigate to your project folder and run:

```sh
python -m venv venv
```
This creates a virtual environment named venv/ inside your project.

#### **2. Activate the Virtual Environment**

Ensure Python 3.8+ is installed. Then install the required libraries:


```sh
# On macOS/ Linux
source venv/bin/activate

## on Windows
# venv\Scripts\activate
```

You should see (venv) appear in your terminal prompt, indicating the virtual environment is active.

#### **3. Install Dependencies**

Ensure Python 3.8+ is installed. Then install the required libraries:

```sh
pip install -r requirements.txt
```

#### **4. Run the Project**

```sh
python main.py
```
Everything will now run inside the virtual environment.

This will generate:
```
output/matched_results_train.csv
output/matched_results_test.csv  (matches `sample_submission.csv` format)
```

#### **5. Deactivate Virtual Environment**

```sh
deactivate
```

### ** Option 2: Run Using Docker (Recommended for Reproducibility)**

#### **1.  Build the Docker Image**

```sh
docker build -t company-matching .
```

#### **2. Run the Container**


```sh
docker run -v $(pwd)/output:/app/output company-matching
```

The results will be saved in output/.

#### **3. Check Output**

```sh
ls output/
cat output/STest_final_predictions.csv | head -n 10
```

## Handling New Data

If you receive new datasets, follow these steps:
1.	Replace data/STrain.csv, data/STest.csv, or data/G.csv.
2.	Re-run the project:
```sh
python main.py  # For virtual environment
```
OR
```sh
docker run -v $(pwd)/output:/app/output company-matching  # For Docker
```


## Evaluation Criteria

| Error Type                                 | Penalty |
|--------------------------------------------|---------|
| Correct Match                           | 0       |
| Matched to the Wrong Company            | +5      |
| Incorrectly Marked as Not Found (-1)    | +1      |

---
## Contact

For any issues, please reach out before the interview. 
