import pandas as pd
import matplotlib.pyplot as plt
from alerce.core import Alerce
import numpy as np
import os
from scipy.optimize import curve_fit

def plot_light_curve(object_id: str, alerce_client: Alerce, r_band_only: bool = True, 
                            save_path: str = None, show_plot: bool = True, r_squared_from_csv: float = None):
    """
    Plot light curve (MJD vs magnitude) for a single object.
    
    Parameters:
    -----------
    object_id : str
        Object ID to plot
    alerce_client : Alerce
        Initialized Alerce client
    r_band_only : bool, default True
        If True, only plot r-band (fid=2) data.
    save_path : str, optional
        If provided, save plot to this path instead of showing it
    show_plot : bool, default True
        If True, display the plot. If False and save_path is None, creates plot but doesn't show.
    r_squared_from_csv : float, optional
        R² value from CSV file (from curve_filter.py). If provided, will be used instead of calculating.
    
    Returns:
    --------
    bool: True if successful, False otherwise
    """
    
    try:
        # Fetch light curve data from Alerce
        light_curve_df = alerce_client.query_detections(object_id, format="pandas")
        
        if light_curve_df is None or light_curve_df.empty:
            print(f"  No data found for {object_id}")
            return False
    
        if 'magpsf' in light_curve_df.columns:
            mag_col = 'magpsf'
        else:
            print(f"  Error: No valid magnitude column found for {object_id}")
            return False
        
        # Filter to r-band (fid=2) only
        if r_band_only:
            plot_data = light_curve_df[light_curve_df['fid'] == 2].copy()
            if plot_data.empty:
                print(f"  No r-band (fid=2) data found for {object_id}")
                return False
        else:
            plot_data = light_curve_df.copy()
        
        # Sort by MJD
        plot_data = plot_data.sort_values('mjd').reset_index(drop=True)
        
        # Remove any NaN values in magnitude or MJD
        plot_data = plot_data.dropna(subset=['mjd', mag_col])
        
        if plot_data.empty:
            print(f"  No valid data points after filtering NaN values for {object_id}")
            return False
        
        # Convert magnitude to flux for peak finding and fitting
        # flux = 10^(-0.4 * mag)
        plot_data['flux'] = 10**(-0.4 * plot_data[mag_col])
        
        # Find peak (maximum flux = brightest point = minimum magnitude)
        peak_idx = plot_data['flux'].idxmax()
        peak_time = plot_data.loc[peak_idx, 'mjd']
        peak_mag = plot_data.loc[peak_idx, mag_col]
        peak_flux = plot_data.loc[peak_idx, 'flux']
        
        # Get data after peak for decay fitting
        decay_data = plot_data[plot_data['mjd'] > peak_time].copy()
        
        # Initialize fit parameters
        fit_success = False
        r_squared = r_squared_from_csv  # Use CSV value if provided (will be overwritten if we calculate)
        fit_times = None
        fit_flux = None  # Store fit flux for plotting
        calculated_r_squared = None
        
        # Print peak info
        print(f"  Peak at MJD {peak_time:.2f} (mag {peak_mag:.2f})")
        if r_squared_from_csv is not None:
            print(f"  Using R² from CSV: {r_squared_from_csv:.4f}")
        
        # Try to fit t^(-5/3) decay if we have enough data (either for visualization or calculation)
        if len(decay_data) >= 3:
            try:
                # Convert time to days since peak
                decay_data['days_since_peak'] = decay_data['mjd'] - peak_time
                
                # Remove any points with days_since_peak <= 0 (shouldn't happen, but just in case)
                decay_data = decay_data[decay_data['days_since_peak'] > 0]
                
                if len(decay_data) >= 3:
                    # Define the t^(-5/3) power law function
                    def power_law(t, A, t0):
                        """A * (t + t0)^(-5/3)"""
                        return A * np.maximum(t + t0, 1e-10)**(-5/3)
                    
                    # Fit the power law to flux data
                    popt, _ = curve_fit(power_law, 
                                       decay_data['days_since_peak'], 
                                       decay_data['flux'], 
                                       p0=[peak_flux, 1.0],
                                       bounds=([0, 0], [np.inf, np.inf]),
                                       maxfev=2000)
                    
                    # Generate fitted curve for plotting
                    # Start from peak (t=0) and extend to cover decay period
                    max_days = decay_data['days_since_peak'].max()
                    fit_days = np.linspace(0, max_days * 1.1, 200)
                    fit_flux = power_law(fit_days, *popt)
                    
                    # Handle any potential zero or negative flux (shouldn't happen, but be safe)
                    fit_flux = np.maximum(fit_flux, 1e-10)
                    fit_times = peak_time + fit_days
                    
                    # Fit was successful, mark it
                    fit_success = True
                    
                    # Calculate R² only if not provided from CSV (for comparison/info)
                    if r_squared_from_csv is None:
                        y_pred = power_law(decay_data['days_since_peak'], *popt)
                        ss_res = np.sum((decay_data['flux'] - y_pred) ** 2)
                        ss_tot = np.sum((decay_data['flux'] - np.mean(decay_data['flux'])) ** 2)
                        
                        if ss_tot > 0:
                            calculated_r_squared = 1 - (ss_res / ss_tot)
                            r_squared = calculated_r_squared
                            print(f"  Fitted t^(-5/3) decay: R² = {r_squared:.4f}")
                    else:
                        # CSV R² provided, curve fitted for visualization only
                        # r_squared already set to r_squared_from_csv above
                        pass
                    
            except Exception as e:
                if r_squared_from_csv is None:
                    print(f"  Could not fit t^(-5/3) decay: {e}")
                else:
                    print(f"  Note: Could not fit curve for visualization (using CSV R²): {e}")
                    fit_success = False
        
        # Print info if no decay data or fit failed
        if not fit_success and len(decay_data) < 3:
            if len(decay_data) == 0:
                print(f"  No decay data after peak (peak is at last data point)")
            else:
                print(f"  Insufficient decay data for fitting ({len(decay_data)} points, need 3+)")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot all data points
        plt.scatter(plot_data['mjd'], plot_data['flux'], 
                    alpha=0.6, s=20, color='red', label=f'r-band ({len(plot_data)} points)')
        
        # Mark the peak point (in flux)
        plt.scatter(peak_time, peak_flux, 
                   s=200, color='gold', marker='*', 
                   edgecolors='black', linewidths=1.5,
                   zorder=5, label='Peak')
        
        # Plot the fitted t^(-5/3) decay curve if fit was successful
        if fit_success and fit_times is not None and fit_flux is not None:
            if r_squared is not None:
                plt.plot(fit_times, fit_flux, 
                        'b--', linewidth=2, alpha=0.8, 
                        label=f't^(-5/3) decay fit (R² = {r_squared:.3f})')
            else:
                plt.plot(fit_times, fit_flux, 
                        'b--', linewidth=2, alpha=0.8, 
                        label='t^(-5/3) decay fit')
        
        # Set automatic axis limits with padding to ensure data is well-framed
        ax = plt.gca()
        
        # Get data ranges (including fitted curve if present)
        x_min = plot_data['mjd'].min()
        x_max = plot_data['mjd'].max()
        y_min = plot_data['flux'].min()
        y_max = plot_data['flux'].max()
        
        # Include fitted curve in range if present
        if fit_success and fit_times is not None and fit_flux is not None:
            x_min = min(x_min, fit_times.min())
            x_max = max(x_max, fit_times.max())
            y_min = min(y_min, fit_flux.min())
            y_max = max(y_max, fit_flux.max())
        
        # Calculate padding as percentage of range
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Add padding: 5% on each side for x-axis, 10% on each side for y-axis
        # Handle edge cases where range might be very small or zero
        if x_range > 0:
            x_padding = x_range * 0.05
        else:
            # If all x values are the same, add small padding
            x_padding = max(abs(x_min) * 0.05, 1.0) if x_min != 0 else 1.0
        
        if y_range > 0:
            y_padding = y_range * 0.10
        else:
            # If all y values are the same, add small padding
            y_padding = max(abs(y_max) * 0.10, y_max * 0.01) if y_max > 0 else 1e-10
        
        # Set axis limits with padding
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        # For y-axis, ensure we don't go below 0 (flux can't be negative)
        y_min_lim = max(0, y_min - y_padding)
        ax.set_ylim(y_min_lim, y_max + y_padding)
        
        # Labels and title
        plt.xlabel('MJD (Modified Julian Date)', fontsize=12)
        plt.ylabel('Flux', fontsize=12)
        plt.title(f'Light Curve for {object_id}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add some statistics to the plot
        stats_text = f'Points: {len(plot_data)}\n'
        stats_text += f'MJD range: {plot_data["mjd"].min():.2f} - {plot_data["mjd"].max():.2f}\n'
        stats_text += f'Flux range: {plot_data["flux"].min():.6e} - {plot_data["flux"].max():.6e}\n'
        stats_text += f'Peak: MJD {peak_time:.2f}, flux {peak_flux:.6e}'
        if r_squared is not None:
            if r_squared_from_csv is not None:
                stats_text += f'\nR² (from CSV): {r_squared:.3f}'
            else:
                stats_text += f'\nDecay fit R²: {r_squared:.3f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot to {save_path}")
        elif show_plot:
            plt.show()
        else:
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"  Error processing {object_id}: {str(e)}")
        plt.close()  # Make sure to close the figure even on error
        return False


def get_all_OIDs_from_csv(csv_file: str, r_band_only: bool = True, save_dir: str = None, 
                       max_objects: int = None, show_plot: bool = True):
    """
    Loop through all unique OIDs in the CSV file and plot their light curves.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with 'oid' column
    r_band_only : bool, default True
        If True, only plot r-band (fid=2) data.
    save_dir : str, optional
        If provided, save all plots to this directory instead of showing them.
        If None and show_plot=True, plots are displayed sequentially.
    max_objects : int, optional
        Maximum number of objects to process. If None, processes all.
    show_plot : bool, default True
        If True and save_dir is None, display plots sequentially (user closes each before next).
        If False and save_dir is None, creates plots but doesn't show them.
    
    Returns:
    --------
    dict: Statistics about the processing (successful, failed, skipped counts)
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get unique OIDs and create mapping to r_squared from CSV
    unique_oids = df['oid'].unique()
    total_oids = len(unique_oids)
    
    # Create mapping from OID to r_squared (use first value if multiple rows per OID)
    r_squared_map = {}
    if 'r_squared' in df.columns:
        for oid in unique_oids:
            oid_data = df[df['oid'] == oid]
            # Get first non-null r_squared value, or None if all are null
            r_sq_values = oid_data['r_squared'].dropna()
            if len(r_sq_values) > 0:
                # Check if it's "N/A" string or numeric
                first_val = r_sq_values.iloc[0]
                if isinstance(first_val, str) and first_val.strip().upper() == 'N/A':
                    r_squared_map[oid] = None
                else:
                    try:
                        r_squared_map[oid] = float(first_val)
                    except (ValueError, TypeError):
                        r_squared_map[oid] = None
            else:
                r_squared_map[oid] = None
        print(f"Found r_squared column in CSV. Will use CSV values instead of calculating.")
    else:
        print(f"No r_squared column found in CSV. Will calculate R² if possible.")
    
    if max_objects is not None:
        unique_oids = unique_oids[:max_objects]
        print(f"Processing {len(unique_oids)} objects (limited from {total_oids} total)...")
    else:
        print(f"Processing {len(unique_oids)} unique objects...")
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving plots to: {save_dir}")
        show_plot = False  # Don't show if saving
    
    # Initialize Alerce client
    alerce_client = Alerce()
    
    # Statistics
    stats = {
        'total': len(unique_oids),
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Loop through all OIDs
    for idx, object_id in enumerate(unique_oids, 1):
        print(f"\n[{idx}/{len(unique_oids)}] Processing {object_id}...")
        
        # Determine save path if saving
        save_path = None
        if save_dir:
            # Create safe filename from OID
            safe_filename = object_id.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(save_dir, f"{safe_filename}_light_curve.png")
        
        # Get r_squared from CSV if available
        r_squared_from_csv = r_squared_map.get(object_id)
        
        # Plot the light curve
        success = plot_light_curve(object_id, alerce_client, r_band_only=r_band_only,
                                         save_path=save_path, show_plot=show_plot,
                                         r_squared_from_csv=r_squared_from_csv)
        
        if success:
            stats['successful'] += 1
        else:
            stats['failed'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total objects: {stats['total']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"{'='*60}")
    
    return stats