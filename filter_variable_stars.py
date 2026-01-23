from alerce.core import Alerce
import pandas as pd
import os

def filter_variable_stars_and_probabilities(myDF):
    # Initialize the Alerce client
    alerce = Alerce()

    # Read the CSV file
    myDF = pd.read_csv(myDF)

    # Variable star classes
    VARIABLE_STAR_CLASSES = ['LPV']

    # TDE classes
    TDE_CLASSES = ['TDE', 'AGN', 'QSO']

    # Variable star probability threshold
    VARIABLE_STAR_PROBABILITY_THRESHOLD = 0.5

    # TDE probability threshold
    TDE_PROBABILITY_THRESHOLD = 0.05

    # Initialize the columns (set all to False initially)
    myDF['variable_star'] = False
    myDF['tde'] = False
    myDF['variable_star_probability'] = None  # Initialize probability columns
    myDF['tde_probability'] = None

    print(f"Processing {len(myDF)} objects...")
    print(f"Variable star threshold: {VARIABLE_STAR_PROBABILITY_THRESHOLD} ({VARIABLE_STAR_PROBABILITY_THRESHOLD * 100}%)")
    print(f"TDE threshold: {TDE_PROBABILITY_THRESHOLD} ({TDE_PROBABILITY_THRESHOLD * 100}%)")
    print("-" * 60)

    # Loop through all OIDs and check for variable stars and TDEs
    for idx, oid in enumerate(myDF['oid'], 1):
        try:
            # Query the probabilities for the OID
            probabilities = alerce.query_probabilities(oid)  # Returns a LIST
            
            # Iterate through each classification in the list
            for classification in probabilities:
                class_name = classification.get('class_name', '')
                probability = classification.get('probability', 0.0)
                
                # Check for variable star
                if class_name in VARIABLE_STAR_CLASSES and probability >= VARIABLE_STAR_PROBABILITY_THRESHOLD:
                    myDF.loc[myDF['oid'] == oid, 'variable_star'] = True
                    myDF.loc[myDF['oid'] == oid, 'variable_star_probability'] = probability
                    print(f"[{idx}/{len(myDF)}] {oid}: Variable star ({class_name}, P={probability:.4f})")
                    break  # Found variable star, move to next OID
                
                # Check for TDE
                if class_name in TDE_CLASSES and probability >= TDE_PROBABILITY_THRESHOLD:
                    myDF.loc[myDF['oid'] == oid, 'tde'] = True 
                    myDF.loc[myDF['oid'] == oid, 'tde_probability'] = probability
                    print(f"[{idx}/{len(myDF)}] {oid}: TDE ({class_name}, P={probability:.4f})")
                    break  # Found TDE, move to next OID
        except Exception as e:
            print(f"[{idx}/{len(myDF)}] Error processing {oid}: {e}")
            continue

    # Save the updated DataFrame to CSV
    output_file = myDF  # Overwrite the original file, or use a different name
    myDF.to_csv(output_file, index=False)
    print("-" * 60)
    print(f"Results saved to: {output_file}")
    print(f"TDEs found: {myDF['tde'].sum()}")