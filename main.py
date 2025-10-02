from alerce_client import fetch_and_save_ztf_data

if __name__ == '__main__':
    try:
        # Ask the user for the number of objects to fetch
        num_objects_str = input("Enter the number of ZTF objects to fetch: ")
        num_objects = int(num_objects_str)

        # Ask the user for the beginning and ending datapoint range
        datapoint_1_str = input("Enter the beginning datapoint range: ")
        beginning_datapoint = int(datapoint_1_str)
        datapoint_2_str = input("Enter the ending datapoint range: ")
        ending_datapoint = int(datapoint_2_str)

        fetch_and_save_ztf_data(num_objects, beginning_datapoint=beginning_datapoint, ending_datapoint=ending_datapoint)

    except ValueError:
        print("Invalid input. Please enter a valid integer.")