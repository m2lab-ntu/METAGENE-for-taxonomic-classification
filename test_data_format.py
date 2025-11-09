"""
Quick test to verify data format matches the configuration regex.
"""

import re

# Regex from config
header_regex = r"^lbl\|(?P<class_id>\d+)\|(?P<tax_id>\d+)?\|(?P<readlen>\d+)?\|(?P<name>[^/\s]+)(?:/(?P<mate>\d+))?$"
pattern = re.compile(header_regex)

# Test with sample headers from your data
test_headers = [
    "lbl|85|301|45|Pseudomonas-61537/2",
    "lbl|1877|158898|286|Gordonia-1231/2",
    "lbl|429|1502|189|Clostridium-53/1",
    "lbl|152|539|80|Eikenella-41289/2",
    "lbl|1818|147047|127|Desulfovibrio-778/2",
]

print("=" * 80)
print("Testing data format compatibility")
print("=" * 80)
print(f"\nRegex: {header_regex}")
print("\n" + "-" * 80)

all_passed = True
for header in test_headers:
    match = pattern.match(header)
    if match:
        print(f"✓ PASS: {header}")
        print(f"  class_id={match.group('class_id')}, "
              f"tax_id={match.group('tax_id')}, "
              f"readlen={match.group('readlen')}, "
              f"name={match.group('name')}, "
              f"mate={match.group('mate')}")
    else:
        print(f"✗ FAIL: {header}")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✓ All test headers match the regex!")
    print("\n✓ Your data format is compatible with the configuration.")
else:
    print("✗ Some headers don't match!")
    print("\n⚠ You may need to adjust the header_regex in the config.")
print("=" * 80)

# Check actual data file
print("\nChecking first 10 headers from actual data file...")
print("-" * 80)

fasta_path = "/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa"
try:
    with open(fasta_path, 'r') as f:
        count = 0
        matches = 0
        for line in f:
            if line.startswith('>'):
                count += 1
                header = line[1:].strip()  # Remove '>' and whitespace
                match = pattern.match(header)
                if match:
                    matches += 1
                    if count <= 5:
                        print(f"✓ {header}")
                else:
                    if count <= 5:
                        print(f"✗ {header}")
                
                if count >= 100:  # Check first 100 headers
                    break
    
    print(f"\n{matches}/{count} headers matched ({100*matches/count:.1f}%)")
    if matches == count:
        print("✓ All checked headers are valid!")
    else:
        print(f"⚠ {count - matches} headers failed to match")
        
except Exception as e:
    print(f"Error reading file: {e}")

print("=" * 80)

