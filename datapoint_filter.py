import pandas as pd
from alerce.core import Alerce
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from tqdm import tqdm
import signal
import sys

# Parallelization settings
MAX_WORKERS = 20  # Number of concurrent requests (reduced to avoid 403 rate limit errors)
RATE_LIMIT = 0.05  # Seconds between requests per worker (increased to avoid rate limiting)

def fetch_light_curve(ztf_object_id: str, beginning_datapoint: int, ending_datapoint: int, 
                      verbose: bool = False) -> dict | None:
    """
    Fetches the light curve (detections) for a given ZTF object ID from the ALeRCE API.
    Creates its own Alerce client for thread safety.

    Args:
        ztf_object_id (str): The ZTF identifier for the transient object (e.g., "ZTF23aabrisv").
        beginning_datapoint (int): Minimum number of datapoints required.
        ending_datapoint (int): Maximum number of datapoints allowed.
        verbose (bool): If True, prints progress messages.

    Returns:
        A dictionary containing oid and num_detections if the object is found and within range,
        otherwise None.
    """
    if verbose:
        print(f"-> Fetching data for {ztf_object_id} from ALeRCE...")
    try:
        # Create a new client for thread safety (each thread needs its own)
        client = Alerce()
        # Query for all detections for the given object ID
        light_curve_df = client.query_detections(ztf_object_id, format="pandas")

        if light_curve_df is not None and not light_curve_df.empty:
            datapoint_count = len(light_curve_df)
            
            if verbose:
                print(f"--> Found {datapoint_count} datapoints for {ztf_object_id}")
            
            # Check if this object has the right number of datapoints
            if datapoint_count >= beginning_datapoint and datapoint_count <= ending_datapoint:
                if verbose:
                    print(f"{ztf_object_id} has {datapoint_count} datapoints (within range {beginning_datapoint}-{ending_datapoint})")
                return {"oid": ztf_object_id, "num_detections": datapoint_count}
            else:
                if verbose:
                    print(f"{ztf_object_id} has {datapoint_count} datapoints (outside range {beginning_datapoint}-{ending_datapoint})")
                return None
            
        else:
            if verbose:
                print(f"--> No data found for {ztf_object_id} in ALeRCE.")
            return None

    except Exception as e:
        if verbose:
            print(f"[ERROR] An error occurred while fetching data from ALeRCE for {ztf_object_id}: {e}")
        return None

def _fetch_single_light_curve(oid: str, beginning_datapoint: int, ending_datapoint: int) -> Optional[Dict]:
    """Helper function to fetch light curve for a single OID (used in parallel execution)."""
    result = fetch_light_curve(oid, beginning_datapoint, ending_datapoint, verbose=False)
    return result

async def _fetch_light_curves_async(oids: List[str], beginning_datapoint: int, ending_datapoint: int,
                                     max_workers: int = 20, rate_limit: float = 0.1) -> List[Dict]:
    """Fetch light curves for multiple OIDs in parallel using ThreadPoolExecutor."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        loop = asyncio.get_running_loop()
        
        # Create a semaphore to limit concurrent requests (rate limiting)
        semaphore = asyncio.Semaphore(max_workers)
        
        async def fetch_with_rate_limit(oid: str) -> Optional[Dict]:
            async with semaphore:
                # Run the sync function in the thread pool
                result = await loop.run_in_executor(
                    executor, 
                    _fetch_single_light_curve, 
                    oid, 
                    beginning_datapoint, 
                    ending_datapoint
                )
                # Rate limiting
                await asyncio.sleep(rate_limit)
                return result
        
        # Create all tasks
        tasks = [fetch_with_rate_limit(oid) for oid in oids]
        
        # Execute with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                        desc="Fetching light curves"):
            result = await coro
            if result is not None:
                results.append(result)
        
        return results
    finally:
        executor.shutdown(wait=True)

def _save_progress_to_csv(all_light_curves: List[Dict], output_filename: str, skipped_count: int = 0) -> None:
    """
    Helper function to save progress to CSV file.
    Called when process completes or is interrupted.
    """
    if not all_light_curves:
        print("\n⚠️  No data to save.")
        return
    
    try:
        combined_df = pd.DataFrame(all_light_curves)
        file_exists = os.path.exists(output_filename)
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} duplicate OID(s) during fetching.")
        
        if file_exists:
            # Append to existing file (duplicates already filtered during fetching)
            combined_df.to_csv(output_filename, mode='a', header=False, index=False)
            print(f"\n✅ Successfully added {len(combined_df)} new unique object(s) to '{output_filename}'.")
            if len(combined_df) <= 20:  # Only print OIDs if list is small
                print(f"Added OIDs: {combined_df['oid'].tolist()}")
        else:
            # Create new file with header
            combined_df.to_csv(output_filename, index=False)
            print(f"\n✅ Successfully saved {len(all_light_curves)} objects to '{output_filename}'.")
            print(f"New file created. Total rows: {len(combined_df)}")
    except Exception as e:
        print(f"\n❌ Error saving progress to CSV: {e}")
        import traceback
        traceback.print_exc()

def fetch_and_save_ztf_data(n_objects: int, output_filename: str = 'ztf_objects_summary.csv', 
beginning_datapoint: int = 10, ending_datapoint: int = 300, 
max_workers: int = MAX_WORKERS, rate_limit: float = RATE_LIMIT):
    """
    Automates the process of fetching light curves for a given number of ZTF objects
    and saves the combined data to a CSV file with only oid and num_detections columns.
    Uses efficient batching to query 500 objects at a time and parallel processing to speed up.
    Stops when target is reached.

    Args:
        n_objects (int): The number of ZTF objects to fetch.
        output_filename (str): The name of the CSV file to save the data.
        beginning_datapoint (int): The beginning datapoint range.
        ending_datapoint (int): The ending datapoint range.
        max_workers (int): Number of parallel workers for fetching light curves.
        rate_limit (float): Rate limit delay in seconds between requests per worker.
    """
    if n_objects <= 0:
        print("Please provide a positive number of objects to fetch.")
        return

    # Read existing OIDs from CSV if it exists
    existing_oids = set()
    if os.path.exists(output_filename):
        try:
            existing_df = pd.read_csv(output_filename)
            if 'oid' in existing_df.columns:
                existing_oids = set(existing_df['oid'].values)
                print(f"Found {len(existing_oids)} existing OIDs in CSV. Will skip duplicates.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")
            print("Proceeding without duplicate check...")

    alerce_client = Alerce()
    all_light_curves = []
    page_num = 1
    batch_size = 500  # Query 500 objects at a time for efficiency
    skipped_count = 0
    max_retries = 3  # Maximum number of retries for 504/timeout errors
    retry_delay = 5  # Initial delay in seconds
    
    print(f"Starting efficient query for {n_objects} unique ZTF objects (querying {batch_size} at a time)...")
    print(f"Using {max_workers} parallel workers with {rate_limit}s rate limit")
    print("\n💡 Tip: Press Ctrl+C to cancel and save progress at any time.\n")
    
    try:
        while len(all_light_curves) < n_objects:
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Query for a batch of object IDs
                    if retry_count > 0:
                        print(f"Retrying batch {page_num} (attempt {retry_count + 1}/{max_retries})...")
                        time.sleep(retry_delay * retry_count)  # Exponential backoff
                    else:
                        print(f"Querying batch {page_num} ({batch_size} objects)...")
                    
                    objects_df = alerce_client.query_objects(page_size=batch_size, page=page_num, format="pandas")
                    
                    # Check if the objects DataFrame is empty
                    if objects_df.empty:
                        print("No more objects found. The ALeRCE API might be temporarily unavailable or we've reached the end of available objects.")
                        success = True  # Not an error, just no more data
                        break
                    
                    # Convert the object IDs to a list
                    object_ids = objects_df['oid'].tolist()
                    print(f"Retrieved {len(object_ids)} object IDs from batch {page_num}")

                    # Filter out duplicates before processing
                    oids_to_process = [oid for oid in object_ids if oid not in existing_oids]
                    
                    if oids_to_process:
                        # Check if we've already reached our target
                        if len(all_light_curves) >= n_objects:
                            print(f"Target reached! Found {len(all_light_curves)} unique objects.")
                            success = True
                            break
                        
                        # Calculate how many more we need
                        remaining_needed = n_objects - len(all_light_curves)
                        
                        # Limit to what we need (if batch is larger than needed)
                        if len(oids_to_process) > remaining_needed:
                            oids_to_process = oids_to_process[:remaining_needed]
                        
                        # Process objects in parallel
                        print(f"Processing {len(oids_to_process)} objects in parallel...")
                        try:
                            # Run async function
                            batch_results = asyncio.run(
                                _fetch_light_curves_async(
                                    oids_to_process, 
                                    beginning_datapoint, 
                                    ending_datapoint,
                                    max_workers=max_workers,
                                    rate_limit=rate_limit
                                )
                            )
                        except RuntimeError:
                            # If there's already an event loop running, create a new one
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            batch_results = loop.run_until_complete(
                                _fetch_light_curves_async(
                                    oids_to_process, 
                                    beginning_datapoint, 
                                    ending_datapoint,
                                    max_workers=max_workers,
                                    rate_limit=rate_limit
                                )
                            )
                            loop.close()
                        
                        # Add successful results
                        for light_curve_data in batch_results:
                            if light_curve_data is not None:
                                all_light_curves.append(light_curve_data)
                                existing_oids.add(light_curve_data['oid'])  # Add to set to avoid duplicates
                        
                        print(f"Progress: {len(all_light_curves)}/{n_objects} unique objects found")
                    
                    # Count skipped duplicates
                    batch_skipped = len(object_ids) - len(oids_to_process)
                    skipped_count += batch_skipped
                    if batch_skipped > 0:
                        print(f"Skipped {batch_skipped} duplicate OIDs in this batch (total skipped: {skipped_count})")
                    
                    success = True  # Successfully processed this batch
                    
                except Exception as e:
                    error_str = str(e)
                    # Check for 403 Forbidden (rate limiting)
                    is_403_error = '403' in error_str or 'Forbidden' in error_str
                    # Check if it's a 504 or 5xx server error (timeout/server issue)
                    is_server_error = '504' in error_str or ('50' in error_str and not is_403_error) or 'timeout' in error_str.lower() or 'gateway' in error_str.lower()
                    
                    # Handle 403 Forbidden (rate limiting) with longer backoff
                    if is_403_error:
                        if retry_count < max_retries:
                            retry_count += 1
                            # Exponential backoff: wait longer for rate limit errors
                            wait_time = retry_delay * (2 ** retry_count)  # Exponential: 5s, 10s, 20s
                            print(f"[WARNING] Rate limit error (403 Forbidden) on page {page_num}: {e}")
                            print(f"Rate limiting detected - waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                            print(f"💡 Tip: If this persists, reduce MAX_WORKERS or increase RATE_LIMIT in the code.")
                            time.sleep(wait_time)
                        else:
                            print(f"[ERROR] Rate limit error (403) - max retries ({max_retries}) reached.")
                            print(f"💡 Solution: Reduce MAX_WORKERS from {max_workers} to 5, or increase RATE_LIMIT from {rate_limit}s to 0.5s")
                            break
                    # Handle other server errors (504, timeouts)
                    elif is_server_error and retry_count < max_retries - 1:
                        retry_count += 1
                        print(f"[WARNING] Server error (504/timeout) on page {page_num}: {e}")
                        print(f"Waiting {retry_delay * retry_count} seconds before retry {retry_count}/{max_retries}...")
                        time.sleep(retry_delay * retry_count)
                    else:
                        # Non-retryable error or max retries reached
                        print(f"[ERROR] Failed to query object IDs on page {page_num}: {e}")
                        if is_server_error:
                            print(f"Max retries ({max_retries}) reached for this page.")
                            break  # Break out of retry loop, will be handled below
                        else:
                            print("Please check the ALeRCE API status or try again later.")
                            _save_progress_to_csv(all_light_curves, output_filename, skipped_count)
                            print(f"Progress saved to CSV: {output_filename}")
                            return
            
            # Check if we've reached our target before continuing
            if len(all_light_curves) >= n_objects:
                print(f"Target reached! Found {len(all_light_curves)} unique objects.")
                break
            
            if success:
                # Only increment page number if we successfully processed it
                page_num += 1
            elif retry_count >= max_retries:
                # If we couldn't process this page after retries, move to next page
                print(f"Skipping page {page_num} after {max_retries} failed attempts. Moving to next page...")
                page_num += 1
                continue
            else:
                # Non-retryable error, exit
                break

    except KeyboardInterrupt:
        # User pressed Ctrl+C - save progress and exit gracefully
        print("\n\n⚠️  Process interrupted by user (Ctrl+C).")
        print(f"📊 Progress: {len(all_light_curves)}/{n_objects} objects found so far.")
        print("💾 Saving progress to CSV...")
        _save_progress_to_csv(all_light_curves, output_filename, skipped_count)
        print("\n✅ Progress saved! Exiting gracefully.")
        return
    
    # Normal completion - save all data
    if all_light_curves:
        # Limit to exactly n_objects if we found more than needed
        if len(all_light_curves) > n_objects:
            all_light_curves = all_light_curves[:n_objects]
        
        _save_progress_to_csv(all_light_curves, output_filename, skipped_count)
    else:
        print(f"\nNo valid unique objects found matching the criteria after skipping {skipped_count} duplicates.")