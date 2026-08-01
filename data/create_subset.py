import json
import random
from collections import defaultdict

def create_balanced_subset(input_file, output_file, total_pairs=1000):
    categories = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cat = data.get('category', 'Unknown')
            categories[cat].append(line)
            
    num_categories = len(categories)
    if num_categories == 0:
        print("No valid categories found in the file.")
        return

    base_count = total_pairs // num_categories
    remainder = total_pairs % num_categories

    subset = []
    
    for cat, items in categories.items():
        sample_size = base_count + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
            
        sample_size = min(sample_size, len(items))
        subset.extend(random.sample(items, sample_size))
        
    random.shuffle(subset)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in subset:
            f.write(item + '\n')
            
    print(f"Generated {len(subset)}pairs to {output_file}")

if __name__ == "__main__":
    create_balanced_subset('bbq_twins.jsonl', 'subset.jsonl')