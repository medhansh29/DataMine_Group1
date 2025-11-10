import pandas as pd
from alerce.core import Alerce

# --- Configuration ---
alerce = Alerce()

# --- Main Execution ---
if __name__ == '__main__':
    print("-> Querying ALL available classifiers and their versions using alerce.query_classifiers()...")
    
    try:
        # Use the function name provided by the user
        classifier_list = alerce.query_classifiers(format='pandas')
        
        if classifier_list.empty:
            print("❌ Query returned no classifier versions.")
        else:
            print("\n--- ✅ Available Classifiers and Versions in ALeRCE ---")
            print(f"Total classifier versions found: {len(classifier_list)}")
            
            # Print the full DataFrame
            print(classifier_list.to_string())
            
            # Print unique classifier names for easier identification
            print("\nUnique Classifier Names:")
            print(classifier_list['classifier_name'].unique().tolist())
            
    except Exception as e:
        print(f"❌ Query FAILED with error: {e}")