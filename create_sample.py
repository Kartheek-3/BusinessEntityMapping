import os
import pandas as pd

os.makedirs('data/raw/train', exist_ok=True)

# s1
s1 = pd.DataFrame([
    {'entity_id': 'S1-1', 'business_name': 'Acme Corp', 'business_address': '123 Main St', 'country': 'US'},
    {'entity_id': 'S1-2', 'business_name': 'Globex Inc', 'business_address': '456 Elm St', 'country': 'US'},
    {'entity_id': 'S1-3', 'business_name': 'Initech', 'business_address': '789 Oak Ave', 'country': 'US'},
])
s1.to_csv('data/raw/train/train_source1.tsv', sep='\t', index=False)

# s2
s2 = pd.DataFrame([
    {'entity_id': 'S2-1', 'business_name': 'Acme Corporation', 'business_address': '123 Main Street', 'country': 'US'},
    {'entity_id': 'S2-2', 'business_name': 'Umbrella Corp', 'business_address': '999 Zombie Ln', 'country': 'US'},
])
s2.to_csv('data/raw/train/train_source2.tsv', sep='\t', index=False)

# s3
s3 = pd.DataFrame([
    {'entity_id': 'S3-1', 'business_name': 'Globex', 'business_address': '456 Elm Street', 'country': 'US'},
    {'entity_id': 'S3-2', 'business_name': 'Initech LLC', 'business_address': '789 Oak Avenue', 'country': 'US'},
])
s3.to_csv('data/raw/train/train_source3.tsv', sep='\t', index=False)

# gt
gt = pd.DataFrame([
    {'source1_entity_id': 'S1-1', 'matched_entity_ids': 'S2-1'},
    {'source1_entity_id': 'S1-2', 'matched_entity_ids': 'S3-1'},
    {'source1_entity_id': 'S1-3', 'matched_entity_ids': 'S3-2'},
])
gt.to_csv('data/raw/train/train_ground_truth.tsv', sep='\t', index=False)
print('Small sample created.')
