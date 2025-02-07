import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import fuzz, process
from src.utils import standardize_company_name

class CompanyMatcher:
    def __init__(self, G):
        """Initialize matcher with ground-truth dataset G."""
        self.G = G.copy()
        self.G["clean_name"] = self.G["name"].apply(standardize_company_name)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=10_000)
        self.nn = None  # Nearest Neighbor model
        
        # Train TF-IDF Model
        self._train_tfidf()

    def _train_tfidf(self):
        """Train the TF-IDF vectorizer and Nearest Neighbors model."""
        tfidf_g_matrix = self.vectorizer.fit_transform(self.G["clean_name"])
        self.nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)
        self.nn.fit(tfidf_g_matrix)

    def match_companies(self, S, test_mode=False):
        """
        Match S dataset with G dataset using multiple strategies.
        
        Args:
            S (pd.DataFrame): External dataset to be matched.
            test_mode (bool): If True, return only test_index (for STest); otherwise, return train_index and company_id.

        Returns:
            pd.DataFrame: Matched results with company_id_pred and optionally company_id (if in training mode).
        """
        S = S.copy()
        S["clean_name"] = S["name"].apply(standardize_company_name)
        
        # Vectorize and find Nearest Neighbors
        tfidf_s_matrix = self.vectorizer.transform(S["clean_name"])
        distances, best_match_indices = self.nn.kneighbors(tfidf_s_matrix, return_distance=True)
        best_match_scores = 1 - distances

        final_matches = []
        for i in range(len(S)):
            if best_match_scores[i][0] >= 0.85:
                final_matches.append(self.G["clean_name"].iloc[best_match_indices[i][0]])
            else:
                fuzzy_candidates = [self.G["clean_name"].iloc[idx] for idx in best_match_indices[i]]
                fuzzy_match_result = process.extractOne(S["clean_name"].iloc[i], fuzzy_candidates, scorer=fuzz.token_sort_ratio)
                final_matches.append(fuzzy_match_result[0] if fuzzy_match_result and fuzzy_match_result[1] >= 92 else None)

        S["matched_name"] = final_matches
        G_dict = dict(zip(self.G["clean_name"], self.G["company_id"]))
        S["company_id_pred"] = S["matched_name"].map(G_dict).fillna(-1).astype(int)

        if not test_mode:
            return S[["train_index", "name", "company_id", "company_id_pred"]]  # Keep company_id for cost calculation
        else:
            return S[["test_index", "company_id_pred"]]