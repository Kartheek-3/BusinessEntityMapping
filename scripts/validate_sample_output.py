import argparse
import pandas as pd
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-s1", required=True)
    parser.add_argument("--test-s2", required=True)
    parser.add_argument("--test-s3", required=True)
    parser.add_argument("--matching-results", required=True)
    parser.add_argument("--candidate-pairs", required=True)
    args = parser.parse_args()

    # 1. & 2. Files exist
    if not os.path.exists(args.matching_results):
        print("FAIL: matching_results.tsv does not exist")
        sys.exit(1)
    if not os.path.exists(args.candidate_pairs):
        print("FAIL: candidate_pairs.tsv does not exist")
        sys.exit(1)

    s1 = pd.read_csv(args.test_s1, sep="\t", dtype=str)
    s2 = pd.read_csv(args.test_s2, sep="\t", dtype=str)
    s3 = pd.read_csv(args.test_s3, sep="\t", dtype=str)

    # 11. Output is tab-separated
    try:
        mr = pd.read_csv(args.matching_results, sep="\t", dtype=str)
        cp = pd.read_csv(args.candidate_pairs, sep="\t", dtype=str)
    except Exception as e:
        print(f"FAIL: Could not read TSV properly: {e}")
        sys.exit(1)

    # 12. Header names are exactly correct
    if list(mr.columns) != ["source1_entity_id", "matched_entity_ids"]:
        print(f"FAIL: matching_results headers wrong: {list(mr.columns)}")
        sys.exit(1)
    
    if list(cp.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        print(f"FAIL: candidate_pairs headers wrong: {list(cp.columns)}")
        sys.exit(1)

    num_test_s1 = len(s1)
    
    # 3. & 4. Number of rows = number of test Source 1 entities
    if len(mr) != num_test_s1:
        print(f"FAIL: matching_results rows ({len(mr)}) != test S1 rows ({num_test_s1})")
        sys.exit(1)
    
    if len(cp) != num_test_s1:
        print(f"FAIL: candidate_pairs rows ({len(cp)}) != test S1 rows ({num_test_s1})")
        sys.exit(1)

    # 5. No duplicate source1_entity_id
    if mr["source1_entity_id"].duplicated().any():
        print("FAIL: duplicate source1_entity_id in matching_results")
        sys.exit(1)
    if cp["source1_entity_id"].duplicated().any():
        print("FAIL: duplicate source1_entity_id in candidate_pairs")
        sys.exit(1)

    valid_targets = set(s2["entity_id"]).union(set(s3["entity_id"]))
    
    cp_map = {}
    for _, row in cp.iterrows():
        cands = []
        if pd.notna(row["candidate_entity_ids"]) and row["candidate_entity_ids"].strip():
            cands = [x.strip() for x in str(row["candidate_entity_ids"]).split(",")]
        cp_map[row["source1_entity_id"]] = set(cands)

    for _, row in mr.iterrows():
        matches = []
        if pd.notna(row["matched_entity_ids"]) and row["matched_entity_ids"].strip():
            matches = [x.strip() for x in str(row["matched_entity_ids"]).split(",")]
        
        # 6. No duplicate matched IDs inside a row
        if len(matches) != len(set(matches)):
            print(f"FAIL: duplicate matched IDs in row for {row['source1_entity_id']}")
            sys.exit(1)
            
        for m in matches:
            # 7. Every matched ID starts with S2- or S3-
            if not (m.startswith("S2-") or m.startswith("S3-")):
                print(f"FAIL: matched ID {m} does not start with S2- or S3-")
                sys.exit(1)
            
            # 8. Every matched ID exists in test_source2 or test_source3
            if m not in valid_targets:
                print(f"FAIL: matched ID {m} not in test S2 or S3")
                sys.exit(1)
                
            # 9. Every predicted match belongs to candidate_pairs
            if m not in cp_map.get(row["source1_entity_id"], set()):
                print(f"FAIL: matched ID {m} not in candidate_pairs for {row['source1_entity_id']}")
                sys.exit(1)

    print("SUCCESS: All final validation checks passed.")

if __name__ == "__main__":
    main()
