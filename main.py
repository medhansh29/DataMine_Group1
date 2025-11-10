from curve_filter import filter_curve
from datapoint_filter import fetch_and_save_ztf_data
from redshifts import fetch_redshifts_from_csv
from date_filter import fetch_detection_dates_menu_option
from offset import process_offsets_from_csv
from scorer import calculate_and_save_tde_scores
import os
import time

# Resolve CSV relative to this file regardless of CWD
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "ztf_objects_summary.csv")

def show_menu():
    """Display the main menu options"""
    print("\n" + "="*50)
    print("           DataMine Physics Tool")
    print("="*50)
    print("Choose an option:")
    print("1. Fetch new ZTF objects")
    print("2. Fetch redshift data for existing objects in CSV")
    print("3. Fetch first/last detection dates for objects in CSV")
    print("4. Filter curve data for objects in CSV")
    print("5. Compute angular offsets and normalized offsets for objects in CSV")
    print("6. Calculate TDE scores and rank candidates")
    print("7. Exit")
    print("="*50)

if __name__ == '__main__':
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                # Fetch new ZTF objects
                print("\n--- Fetching New ZTF Objects ---")
                num_objects_str = input("Enter the number of ZTF objects to fetch: ")
                num_objects = int(num_objects_str)

                datapoint_1_str = input("Enter the beginning datapoint range: ")
                beginning_datapoint = int(datapoint_1_str)
                datapoint_2_str = input("Enter the ending datapoint range: ")
                ending_datapoint = int(datapoint_2_str)

                start_time = time.time()
                print("Fetching ZTF object data...")
                fetch_and_save_ztf_data(num_objects, beginning_datapoint=beginning_datapoint, ending_datapoint=ending_datapoint)
                print("ZTF object data fetch completed!")
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time taken: {elapsed_time:.2f} seconds")
                
            elif choice == '2':
                # Fetch redshift data for existing objects
                print("\n--- Fetching Redshift Data ---")
                
                # Check if CSV file exists
                if not os.path.exists(CSV_PATH):
                    print("Error: ztf_objects_summary.csv file not found!")
                    print("Please fetch some ZTF objects first (option 1).")
                    continue
                
                print("Fetching redshift data for existing objects...")
                fetch_redshifts_from_csv()
                print("Redshift data fetch completed!")
                
            elif choice == '3':
                # Fetch first/last detection dates
                fetch_detection_dates_menu_option(CSV_PATH)

            elif choice == '4':
                # Filter curve data for objects in CSV
                if not os.path.exists(CSV_PATH):
                    print(f"Error: {os.path.basename(CSV_PATH)} file not found!")
                else:
                    print("\n--- Filtering light curves and computing metrics ---")
                    updated = filter_curve(CSV_PATH)
                    print("Light curve filtering completed and CSV updated.")

            elif choice == '5':
                # Compute angular offsets and normalized offsets
                if not os.path.exists(CSV_PATH):
                    print(f"Error: {os.path.basename(CSV_PATH)} file not found!")
                    print("Please fetch some ZTF objects first (option 1).")
                else:
                    print("\n--- Computing Angular Offsets and Normalized Offsets ---")
                    print("Note: This uses DELIGHT and may take some time...")
                    updated = process_offsets_from_csv(CSV_PATH)
                    if not updated.empty:
                        print("Offset computation completed and CSV updated.")

            elif choice == '6':
                # Calculate TDE scores and rank candidates
                if not os.path.exists(CSV_PATH):
                    print(f"Error: {os.path.basename(CSV_PATH)} file not found!")
                    print("Please fetch some ZTF objects first (option 1).")
                else:
                    print("\n--- Calculating TDE Scores and Ranking Candidates ---")
                    result = calculate_and_save_tde_scores(CSV_PATH)
                    if result is not None:
                        print("TDE scoring completed and CSV updated.")

            elif choice == '7':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, or 7.")
                
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")