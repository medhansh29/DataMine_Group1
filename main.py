from curve_filter import filter_curve
from datapoint_filter import fetch_and_save_ztf_data
from redshifts import fetch_redshifts_from_csv
from date_filter import fetch_detection_dates_menu_option
import os

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
    print("5. Exit")
    print("="*50)

if __name__ == '__main__':
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                # Fetch new ZTF objects
                print("\n--- Fetching New ZTF Objects ---")
                num_objects_str = input("Enter the number of ZTF objects to fetch: ")
                num_objects = int(num_objects_str)

                datapoint_1_str = input("Enter the beginning datapoint range: ")
                beginning_datapoint = int(datapoint_1_str)
                datapoint_2_str = input("Enter the ending datapoint range: ")
                ending_datapoint = int(datapoint_2_str)

                print("Fetching ZTF object data...")
                fetch_and_save_ztf_data(num_objects, beginning_datapoint=beginning_datapoint, ending_datapoint=ending_datapoint)
                print("ZTF object data fetch completed!")
                
            elif choice == '2':
                # Fetch redshift data for existing objects
                print("\n--- Fetching Redshift Data ---")
                
                # Check if CSV file exists
                if not os.path.exists("ztf_objects_summary.csv"):
                    print("Error: ztf_objects_summary.csv file not found!")
                    print("Please fetch some ZTF objects first (option 1).")
                    continue
                
                print("Fetching redshift data for existing objects...")
                fetch_redshifts_from_csv()
                print("Redshift data fetch completed!")
                
            elif choice == '3':
                # Fetch first/last detection dates
                fetch_detection_dates_menu_option("ztf_objects_summary.csv")

            elif choice == '4':
                # Filter curve data for objects in CSV
                filter_curve("test.csv")

            elif choice == '5':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")