#!/usr/bin/env python3

"""
beta_compute.py


To do list/notes:
 -  instead of calib_hdist and zdist max, maybe use ncalib to find the ncalib 
    nearest events? could improve spatial resolution but need to test
 -  Check bootstrapping to make sure it is done correctly
 -  probably don't need to do bootstrap, have enough samples
Last Modified:
    2025-11-25
"""

import numpy as np
import pandas as pd
from tqdm import trange
import time
import os
import pickle as pkl
import inspect

import matplotlib.pyplot as plt


class BetaEstimator:
    def __init__(self, 
        df_records, 
        spectra, 
        f,
        low_window_desired      = (1.0, 5.0),
        high_window_desired     = (15.0, 22.0),
        calib_mag_range         = (1.4, 1.6),
        calib_hdist_max         = 10.0,
        calib_zdist_max         = 2.0,
        calib_n_records_min     = 10,
        mag_corr_method         = 'smoothedspline',
        mag_corr_dM             = 0.2,
        quiet                   = False,
        save_dir                = None,
        compute_uncertainty     = True,
        n_bootstrap             = 1000,
        confidence_level        = 0.95,
        calib_time_filter       = False,
        calib_time_method       = 'interval',
        calib_time_ndays        = 180,
        calib_time_n_nearest    = 25,
        ):
        if not quiet:
            print("Initializing BetaEstimator")
            print("--------------------------")

        excl = ['self', 'df_records', 'spectra', 'f']
        sig = inspect.signature(self.__init__)
        self.input_parameter_names = [k for k in sig.parameters if k not in excl]
        
        # Store parameters as attributes
        self.df_records = df_records.copy()
        self.spectra = spectra
        self.f = f
        self.df = f[1] - f[0]
        self.low_window_desired = low_window_desired
        self.high_window_desired = high_window_desired
        self.calib_mag_range = calib_mag_range
        self.calib_hdist_max = calib_hdist_max
        self.calib_zdist_max = calib_zdist_max
        self.calib_n_records_min = calib_n_records_min
        self.mag_corr_method = mag_corr_method
        self.mag_corr_dM = mag_corr_dM
        self.quiet = quiet
        self.save_dir = save_dir
        self.compute_uncertainty = compute_uncertainty
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
        self.calib_time_filter = calib_time_filter
        self.calib_time_method = calib_time_method
        self.calib_time_ndays = calib_time_ndays
        self.calib_time_n_nearest = calib_time_n_nearest

        # Validate input
        self.validate_input()
        if not self.quiet: self.print_metadata_information()

        # Do some basic processing before computing logbeta
        self.digitize_names()
        # self.compute_eastings_northings()
        self.compute_column_dependencies()

        # Create event/station metadata dicts
        # self.compute_dicts()

        # Calculate actual indices and frequency bands for beta computation
        self.compute_frequency_bands()
        if not self.quiet: self.print_frequency_information()

        # Get indices of calibration events
        self.store_calibration_events()
        if not self.quiet: self.print_calibration_information()

        if not self.quiet:
            print("STATUS:")
            print("BetaEstimator initialized successfully. Run 'compute()' to continue.")
            print("--------------------------------------------------------------------")

    def update_nrec(self):
        self.df_events['nrec'] = self.df_events['_cid'].apply(len)

    def store_calibration_events(self):
        self.calibration_inds = np.where(np.logical_and(
            self.df_records['emag'].values >= self.calib_mag_range[0], 
            self.df_records['emag'].values <= self.calib_mag_range[1]
            ))[0]
        self.metadata_calib = self.df_records.iloc[self.calibration_inds].reset_index(drop=True)
    
    def compute_metadata_dicts(self):

        # make a lookup dict for station locations station_name: (slon, slat, sele)
        self.station_locations = dict(zip(all_sta_catalog['station_name'], zip(all_sta_catalog['slon'], all_sta_catalog['slat'], all_sta_catalog['sele'])))

        # Make a lookup dict for event origins event_name: (elon, elat, edep, edatetime)
        self.event_origins = dict(zip(eq_df['event_name'], zip(eq_df['elon'], eq_df['elat'], eq_df['edep'], eq_df['edatetime'])))


    def compute(self, recompute=False):

        print("Computing results")
        print("-----------------")
        params_filepath = os.path.join(self.save_dir, 'params.logbeta')
        data_filepath = os.path.join(self.save_dir, 'data.logbeta')

        files_exist = os.path.exists(params_filepath) and os.path.exists(data_filepath)

        # If the files don't exist, compute logbeta no matter what        
        if not files_exist:
            print(f"Results not found in {self.save_dir} \nComputing.")
            do_computation = True
        # Otherwise, check if recompute==True
        else:
            # If True, then compute
            if recompute:
                print("Recomputing & overwriting previous results.")
                do_computation = True
            # If False, check if parameters have been changed
            else:
                # Make sure parameters are unchanged
                t0 = time.time()
                keys = [i for i in self.__dict__.keys() if i[:1] != '_']
                current_params = {key: getattr(self, key) for key in self.input_parameter_names}
                # print("Current params: ", current_params)
                # print("Loaded params: ", self.load_parameters())
                loaded_params = self.load_parameters()
                if current_params == loaded_params:
                    print(f"Parameters unchanged from {params_filepath} \nSkipping computation.")
                    do_computation = False
                else:
                    print(f"Parameters changed from {params_filepath} \nRecomputing.")
                    do_computation = True

        # update/create column 'nrec'

        if do_computation:
            self.compute_logbeta()
            self.store_calibration_events()
            # if 'kappa0' not in self.df_records.columns:
            #     self.df_records['kappa0'] = -999.0
            self.compute_dlogbeta()
            self.apply_magnitude_correction(corr_type=self.mag_corr_method)
            
            self.store_calibration_events()
            if not self.quiet: self.print_calibration_information()
            self.update_nrec()
            self.save_parameters()
            self.save_data()
            self.save_fwf()

            return self
        else:
            # evaluate magnitude range?
            t0 = time.time()
            print(f"Loading data & parameters from {self.save_dir}")
            loaded_instance = self.load_data(self.save_dir)
            self.load_parameters()
            print(f"Done. ({time.time()-t0:.2f}s)")
            # self.save_fwf()
            return loaded_instance

    def save_data(self):
        with open(f"{self.save_dir}/data.logbeta", 'wb') as fs:
            pkl.dump(self, fs)
    
    def save_fwf(self):
        if self.compute_uncertainty:
            cols = ['event_name', 'emag', 'elon', 'elat', 'edep', 'nrec', 'dlogbeta', 'dlogbeta_std', 'dlogbeta_lower', 'dlogbeta_upper', 'dlogbeta_corr']
            fmt = '%8i %5.2f %15.11f %15.11f %7.3f %5i %15.11f %12.7e %12.7e %12.7e %15.11f'
        else:
            cols = ['event_name', 'emag', 'elon', 'elat', 'edep', 'nrec', 'dlogbeta', 'dlogbeta_corr']
            fmt = '%8i %5.2f %15.11f %15.11f %7.3f %5i %15.11f %15.11f'
        np.savetxt(f"{self.save_dir}/logbeta.txt", self.df_events[cols].values, fmt=fmt)

    @staticmethod    
    def load_data(save_dir):
        with open(f"{save_dir}/data.logbeta", 'rb') as fs:
            loaded_instance = pkl.load(fs)
        return loaded_instance

    def save_parameters(self):        
        params = {key: getattr(self, key) for key in self.input_parameter_names}
        with open(f"{self.save_dir}/params.logbeta", 'wb') as fs:
            pkl.dump(params, fs)

    def load_parameters(self):
        with open(f"{self.save_dir}/params.logbeta", 'rb') as fs:
            params = pkl.load(fs)
        return params

    def convert_utc_to_posix_ns(self):
        self.df_records['etime'] = self.df_records['edatetime'].apply(lambda x: x.ns)
        # drop edatetime column
        self.df_records.drop(columns=['edatetime'], inplace=True)

    def convert_posix_ns_to_datetime(self):
        self.df_records['edatetime'] = self.df_records['etime'].apply(UTC)
        # drop edatetime column
        self.df_records.drop(columns=['etime'], inplace=True)

    def compute_dlogbeta(self):
        """Correct path and station effects simultaneously (dlogbeta)

        Last Modified:
            2025-11-24
        """

        eids = np.unique(self.df_records['_eid'].values)
        nevents = len(eids)

        # Store NaNs, since we are looping and some might not be computed
        # self.df_records['dlogbeta'] = np.nan # unnecessary?

        if self.compute_uncertainty:
            # self.df_records['dlogbeta_std'] = np.nan    # also unnecessary
            # self.df_records['dlogbeta_lower'] = np.nan  # also unnecessary
            # self.df_records['dlogbeta_upper'] = np.nan  # also unnecessary
            # self.df_records['dlogbeta_median'] = np.nan # also unnecessary
            # self.pair_dep.extend([
            #     'dlogbeta_std', 
            #     'dlogbeta_lower', 
            #     'dlogbeta_upper', 
            #     'dlogbeta_median'])

            # set confidence intervals for uncertainty
            alpha = 1 - self.confidence_level
            lower_percentile = 100 * (alpha / 2)
            upper_percentile = 100 * (1 - alpha / 2)

        # Pre-compute outside the loop
        eid_groups = self.df_records.groupby('_eid')

        # Vectorize depth filter on metadata_calib once
        calib_edep = self.metadata_calib['edep'].values
        calib_elat = self.metadata_calib['elat'].values
        calib_elon = self.metadata_calib['elon'].values
        calib_cid = self.metadata_calib['_cid'].values
        calib_logbeta = self.metadata_calib['logbeta'].values

        calib_event_name = self.metadata_calib['event_name'].values

        # Pre-extract time data if time filtering is enabled
        if self.calib_time_filter:
            calib_etime = self.metadata_calib['etime'].values

        # Pre-extract target event data as dict for faster access
        target_data = {}
        for eid in eids:
            md_t = eid_groups.get_group(eid)
            target_data[eid] = {
                'edep': md_t['edep'].iloc[0],
                'elat': md_t['elat'].iloc[0],
                'elon': md_t['elon'].iloc[0],
                'etime': md_t['etime'].iloc[0],
                'cids': md_t['_cid'].values,
                'logbeta': md_t['logbeta'].values,
                'indices': md_t.index.values  # Store original indices
            }

        # Pre-allocate results arrays for faster assignment
        dlogbeta_results = np.full(len(self.df_records), np.nan)
        if self.compute_uncertainty:
            dlogbeta_std_results = np.full(len(self.df_records), np.nan)
            dlogbeta_lower_results = np.full(len(self.df_records), np.nan)
            dlogbeta_upper_results = np.full(len(self.df_records), np.nan)
            dlogbeta_median_results = np.full(len(self.df_records), np.nan)


        for i in trange(nevents, desc="Computing corrected logbeta"):

            # Choose ith event ID and metadata
            eid = eids[i]
            target = target_data[eid]
            
            # Use pre-grouped data (much faster than boolean indexing)
            md_t = eid_groups.get_group(eid)
            
            # Vectorized depth filter (no array allocation)
            depth_mask = (calib_edep >= target['edep'] - self.calib_zdist_max) & \
                         (calib_edep <= target['edep'] + self.calib_zdist_max)
            
            # Combined filter with channel match
            channel_mask = np.isin(calib_cid, target['cids'])
            combined_mask = depth_mask & channel_mask
            
            # Add time filter if enabled
            if self.calib_time_filter and self.calib_time_method == 'interval':
                # Convert time difference to days
                time_delta_ns = np.abs(calib_etime - target['etime'])
                time_delta_days = (time_delta_ns / 86400.0) / 1E9
                
                time_mask = time_delta_days <= self.calib_time_ndays
                combined_mask = combined_mask & time_mask

            if not combined_mask.any():
                continue
            
            # Apply mask once
            filtered_elat = calib_elat[combined_mask]
            filtered_elon = calib_elon[combined_mask]
            filtered_event_name = calib_event_name[combined_mask]

            # running_calib_mask = combined_mask.copy()
            
            # Compute distances only for filtered events
            dists = _haversine_km(
                np.full(len(filtered_elat), target['elat']),
                np.full(len(filtered_elon), target['elon']),
                filtered_elat,
                filtered_elon
            )
            
            # Distance filter
            dist_mask = dists <= self.calib_hdist_max
            
            if dist_mask.sum() < self.calib_n_records_min:
                continue
            
            # These mask indices are indices of self.metadata_calib
            # that pass H/V distance to target, and are recorded by the same
            # channel as the target
            calib_mask_indices = np.where(combined_mask)[0][dist_mask]
            c_cid = calib_cid[calib_mask_indices]
            c_logbeta = calib_logbeta[calib_mask_indices]
            
            # This is unnecessary since all c_cid_filtered are in target['cids']
            # # Match channels - simplified since we already have the data
            # mask_c = np.isin(c_cid_filtered, target['cids'])
            # c_cid = c_cid_filtered[mask_c]
            # c_logbeta = c_logbeta_filtered[mask_c]

            # assert len(mask_c) == sum(mask_c), 'uh oh'

            # End of depth, channel, and distance filtering. Now, check
            # which records in target['cids'] are in c_cid

            # Toss any record that doesn't have a calibration record at the
            # same channel
            mask_t = np.isin(target['cids'], c_cid)
            t_cid = target['cids'][mask_t]
            t_logbeta = target['logbeta'][mask_t]
            
            # Sort target data by channel ID for efficient matching
            sorter = np.argsort(t_cid)
            t_cid_sorted = t_cid[sorter]
            t_logbeta_sorted = t_logbeta[sorter]

            # Find matching positions for calibration channels in sorted target data
            v = np.searchsorted(t_cid_sorted, c_cid)
            
            # Compute all pairwise differences
            differences = t_logbeta_sorted[v] - c_logbeta

            # Group differences by channel (c_cid) and compute median for each
            unique_channels, inverse_indices = np.unique(c_cid, return_inverse=True)

            # Use reduceat for efficient grouped operations
            sorted_order = np.argsort(inverse_indices)
            sorted_diffs = differences[sorted_order]
            sorted_inverse = inverse_indices[sorted_order]

            # Find split points where channel changes
            split_indices = np.r_[0, np.where(np.diff(sorted_inverse) != 0)[0] + 1, len(sorted_inverse)]

            # Split into channel groups
            channel_groups = [sorted_diffs[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1)]

            # Compute median for each channel group
            channel_medians = np.array([np.median(el) for el in channel_groups], dtype=float)
            channel_counts  = np.array([len(el) for el in channel_groups], dtype=int)

            final_mask = channel_counts >= self.calib_n_records_min

            if final_mask.sum() < 3:
                continue

            channel_medians = channel_medians[final_mask]
            channel_counts = channel_counts[final_mask]

            # print(channel_groups)
            # print(channel_medians)
            # print(channel_counts)

            # plt.figure()
            # plt.scatter(channel_medians, channel_counts, s=1, c='k')
            # plt.show()
            # if i > 20: raise ValueError()

            # Final median of medians
            median_diff = np.median(channel_medians)

            # Assign to pre-allocated array
            dlogbeta_results[target['indices']] = median_diff


            if self.compute_uncertainty:
                # bootstrap_estimates = np.zeros(self.n_bootstrap)
                n_samples = len(channel_medians)
                resample_indices = np.random.choice(
                    n_samples, size=(self.n_bootstrap, n_samples), replace=True)
                bootstrap_estimates = np.median(channel_medians[resample_indices], axis=1)


                dlogbeta_std_results[target['indices']] = np.std(bootstrap_estimates)
                dlogbeta_lower_results[target['indices']] = np.percentile(
                    bootstrap_estimates, lower_percentile)
                dlogbeta_upper_results[target['indices']] = np.percentile(
                    bootstrap_estimates, upper_percentile)
                dlogbeta_median_results[target['indices']] = np.median(bootstrap_estimates)

            # DEBUGGING


        

        # Assign results back to DataFrame once at the end
        self.df_records['dlogbeta'] = dlogbeta_results
        if self.compute_uncertainty:
            self.df_records['dlogbeta_std'] = dlogbeta_std_results
            self.df_records['dlogbeta_lower'] = dlogbeta_lower_results
            self.df_records['dlogbeta_upper'] = dlogbeta_upper_results
            self.df_records['dlogbeta_median'] = dlogbeta_median_results

        # drop rows with NaN dlogbeta
        self.df_records = self.df_records.dropna(subset=['dlogbeta']).reset_index(drop=True)


    def apply_magnitude_correction(self, corr_type='smoothedspline'):
        self.group_events()
        mag_corr_dM = self.mag_corr_dM
        if corr_type == 'smoothedspline':
            from scipy.signal import savgol_filter
            from scipy.stats import binned_statistic

            # Determine the range of magnitude bins: min and max magnitudes
            # rounded to the nearest mag_corr_dM. np.round fixes rounding error
            M_range = np.round((
                np.floor(np.min(self.df_events['emag'].values) / mag_corr_dM) * mag_corr_dM, 
                np.ceil(np.max(self.df_events['emag'].values) / mag_corr_dM) * mag_corr_dM
                ), 4)
            
            # Define the edges and midpoints of the magnitude bins
            edges = np.arange(M_range[0], M_range[1]+1*mag_corr_dM, mag_corr_dM)
            midpoints = (edges[:-1] + edges[1:])/2

            # drop events with emag lower than the first bin (interpolation issue?)
            self.df_events = self.df_events[self.df_events['emag']>=midpoints[0]].reset_index(drop=True)

            # Store magnitudes and dlogbetas
            emag = self.df_events['emag'].values
            dlogbeta = self.df_events['dlogbeta'].values

            # Assign each target event to a magnitude bin
            bin_inds = np.digitize(emag, edges) - 1

            # # Compute the median dlogbeta for each magnitude bin
            # median_dlbeta = np.empty(len(edges)-1) * np.nan
            # # print('computing median_dlbeta')
            # for i in range(len(median_dlbeta)):
            #     median_dlbeta[i] = np.median(self.df_events['dlogbeta'].values[bin_inds==i])

            # Compute the median dlogbeta for each magnitude bin
            median_dlbeta = binned_statistic(emag, dlogbeta, statistic='median', bins=edges)[0]

            filter_window_len = int(np.floor(len(median_dlbeta)/4))
            polyorder = 3
            if polyorder >= filter_window_len:
                polyorder = filter_window_len-1
            naninds = np.isnan(median_dlbeta)
            # print(f"Magnitude range: {M_range[0]:.2f} to {M_range[1]:.2f}")
            # print("Bins (midpoints) with fewer than required earthquakes: ", midpoints[naninds])
            # print(f"Smoothing using Savgol filter with window length {filter_window_len} and polynomial order {polyorder}")
            realmids = midpoints[~naninds]
            median_dlbeta_smooth = savgol_filter(median_dlbeta[~naninds], filter_window_len, polyorder)
            corrections = np.interp(self.df_events['emag'].values, realmids, median_dlbeta_smooth)
            self.correction_mags = realmids
            self.correction_function = median_dlbeta_smooth
            self.df_events['dlogbeta_corr'] = self.df_events['dlogbeta'].values - corrections

            # print("median_dlbeta[~naninds]: ", median_dlbeta[~naninds])
            # print("num nan in emag: ", np.sum(np.isnan(self.df_events['emag'].values)))
            # print("realmids: ", realmids)
            # print("median_dlbeta_smooth: ", median_dlbeta_smooth)
            # print('num nan in corrections: ', np.sum(np.isnan(corrections)))
        
        # self.pair_dep.append('dlogbeta_corr')
        self.explode_events()
        self.group_channels()

    def plot_corrections(self):
        import matplotlib.pyplot as plt
        plt.plot(self.correction_mags, self.correction_function)
        plt.xlabel('Magnitude')
        plt.ylabel('dlogbeta correction')
        plt.show()

    def save_results(
        self, 
        filename="results.txt", 
        fields=['event_name', 'emag', 'elon', 'elat', 'edep', 'dlogbeta', 'dlogbeta_corr'],
        fmt = '%8i %5.2f %15.11f %15.11f %7.3f %15.11f %15.11f'):

        if self.compute_uncertainty:
            fields=['event_name', 'emag', 'elon', 'elat', 'edep', 'dlogbeta', 'dlogbeta_median', 'dlogbeta_std', 'dlogbeta_lower', 'dlogbeta_upper', 'dlogbeta_corr']
        
        np.savetxt(
            filename,
            self.df_events[fields].values,
            fmt=fmt,
            header=' '.join(fields) + '\n'
        )

    def group_events(self):
        print("group_events()")
        # group self.df_records by event dependent fields
        self.compute_column_dependencies()
        # return self
        self.df_events = self.df_records.groupby(self.ev_dep, as_index=False)[self.ch_dep+self.pair_dep].agg(list)

    def group_channels(self):
        print("group_channels()")
        # group self.df_records by channel-dependent fields
        self.compute_column_dependencies()
        self.df_channels = self.df_records.groupby(self.ch_dep, as_index=False, dropna=False)[self.ev_dep+self.pair_dep].agg(list)

    def explode_events(self):
        print('explode_events()')
        self.compute_column_dependencies()
        self.df_records = self.df_events.explode(self.ch_dep+self.pair_dep)

    def explode_channels(self):
        print('explode_channels()')
        self.compute_column_dependencies()
        self.df_records = self.df_channels.explode(self.ev_dep+self.pair_dep)

    def compute_column_dependencies(self):
        # Columns in self.df_records are 'dependent' in one of three ways:
        #     1) event-dependent (e.g. EQ magnitude, EQ location, etc.)
        #     2) channel-dependent (e.g. channel name, station location, etc.)
        #     3) event-channel pair-dependent (e.g. event-station distance, etc.)
        # 
        # This function checks the columns and updates ev_dep, ch_dep, pair_dep
        # class variables.
        ev_dep_columns = [
            'event_name', 'emag', 'elat', 'elon', 'edep', 'etime', 'edatetime', '_eid', 'event_id', 
            'evid', 'event', 'ex', 'ey', 'nts', 'dlogbeta', 'dlogbeta_corr',
            'nrec', 'dlogbeta_std', 'dlogbeta_lower', 'dlogbeta_upper',
            'dlogbeta_median'] # fix nts later
        cha_dep_columns = [
            'channel_name', 'slat', 'slon', 'selev', '_cid', 'sx', 'sy', 'kappa0'
            ]
        pair_dep_columns = [
            'logbeta', 'deldist', 'a2_max', 's1', 's2', 'stn'
            ]
        columns = self.df_records.columns
        self.ev_dep = []
        self.ch_dep = []
        self.pair_dep = []

        for col in columns:
            if col in ev_dep_columns:
                self.ev_dep.append(col)
            elif col in cha_dep_columns:
                self.ch_dep.append(col)
            elif col in pair_dep_columns:
                self.pair_dep.append(col)
            else:
                raise Warning("Column {} not recognized".format(col))


    def validate_input(self):
        columns = self.df_records.columns
        # Rename columns to standardized names. Conventions:
        #     1) event-dependent quantities begin with e-
        #     2) station/channel-dependent quantities begin with s-
        #     3) event_name and channel_name are unique identifiers for
        #        events and channels, respectively

        ### Check column names and verify all required columns are present ###
        # Check for an event_name column
        event_name_columns = ['event_id', 'evid', 'event']
        if 'event_name' not in columns:
            for col in event_name_columns:
                if col in columns:
                    self.df_records.rename(
                        columns={col: 'event_name'}, inplace=True)
                    print("Renamed column {} to event_name".format(col))
                    break
        if "event_name" not in self.df_records.columns:
            raise ValueError("No event_name column found. Check input data.")
        
        # Check for a channel_name column
        channel_name_columns = ['station_id', 'sid', 'station', 'stname', 
                                'st_name', 'stid']
        if 'channel_name' not in columns:
            for col in channel_name_columns:
                if col in columns:
                    self.df_records.rename(
                        columns={col: 'channel_name'}, inplace=True)
                    print("Renamed column {} to channel_name".format(col))
                    break
        if "channel_name" not in self.df_records.columns:
            raise ValueError("No channel_name column found. Check input data.")
        
        # Change qmag, qlat, qlon, qdep into emag, elat, elon, edep
        if 'qmag' in columns:
            self.df_records.rename(columns={'qmag': 'emag'}, inplace=True)
        if 'qlat' in columns:
            self.df_records.rename(columns={'qlat': 'elat'}, inplace=True)
        if 'qlon' in columns:
            self.df_records.rename(columns={'qlon': 'elon'}, inplace=True)
        if 'qdep' in columns:
            self.df_records.rename(columns={'qdep': 'edep'}, inplace=True)

        # Check for required columns
        required_columns = [
            'event_name', 'channel_name', 'emag', 'elat', 'elon', 'edep', 
            'slat', 'slon', 'selev', 'deldist']
        for col in required_columns:
            if col not in self.df_records.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert edatetime to posix ns 
        if self.calib_time_filter:
            self.convert_utc_to_posix_ns()

        ### Check some input data to make sure it's reasonable ###

        # Spectra shape and f array
        if len(self.df_records) != self.spectra.shape[0]:
            raise ValueError("Metadata and spectra row counts do not match.")
        if len(self.f) != self.spectra.shape[1]:
            raise ValueError(
                f"Spectra width {self.spectra.shape[1]} does not match \
                f array length {len(self.f)}.")

        # Deldist should be in km
        if (np.median(self.df_records['deldist']) > 100) or \
            (np.median(self.df_records['deldist']) < 10):
            raise Warning(
                f"Is deldist in km? Median is" \
                f" {np.median(self.df_records['deldist'])}")
        
        # Check for NaNs in spectra
        assert np.sum(np.isnan(self.spectra)) == 0, \
            "Spectra contains NaNs."
        
        # # Spectra should not be logarithmic:
        # qnty = np.median(
        #     np.max(self.spectra, axis=1) / np.min(self.spectra, axis=1)
        #     )
        # if qnty < 10:
        #     raise Warning(
        #         f"Spectra should be linear but appear to be logarithmic. {qnty:.4e}"
        #         )

        # Store some initial counts
        self.nrecords_initial = len(self.df_records)
        self.nevents_initial = len(self.df_records['event_name'].unique())
        self.nchannels_initial = len(self.df_records['channel_name'].unique())

        if self.save_dir is None: self.save_dir = 'tmp/'
        os.makedirs(self.save_dir, exist_ok=True)


    def digitize_names(self):
        # First, sort df_records:
        self.df_records = self.df_records.sort_values(by=['event_name', 'channel_name'])
        v = self.df_records.index.values
        self.df_records = self.df_records.reset_index(drop=True)
        
        # Now, sort the spectra array
        self.spectra = self.spectra[v, :]

        self.df_records, self.event_name_dict = _get_id_from_column(self.df_records, 'event_name', '_eid')
        self.df_records, self.channel_name_dict = _get_id_from_column(self.df_records, 'channel_name', '_cid')

    def compute_logbeta(self):
        # spectra is an (N x nf) array
        low_inds = self.low_window_inds
        high_inds = self.high_window_inds
        
        low_band = np.median(np.log10(self.spectra[:, low_inds[0]:low_inds[1]+1]), axis=1)
        high_band = np.median(np.log10(self.spectra[:, high_inds[0]:high_inds[1]+1]), axis=1)
        self.df_records['logbeta'] = high_band - low_band
        del self.spectra
    
    # def compute_eastings_northings(self):
    #     # this will need to be improved for a larger area
    #     self.df_records['sx'], self.df_records['sy'], zn, zl = utm.from_latlon(
    #         self.df_records['slat'].values, 
    #         self.df_records['slon'].values
    #         )
    #     self.df_records['ex'], self.df_records['ey'], zn, zl = utm.from_latlon(
    #         self.df_records['elat'].values, 
    #         self.df_records['elon'].values
    #         )

    def compute_frequency_bands(self):
        self.low_window_inds = _get_inds_of_values_in_array(self.f, self.low_window_desired)
        self.high_window_inds = _get_inds_of_values_in_array(self.f, self.high_window_desired)

        self.low_window = [self.f[self.low_window_inds[0]], self.f[self.low_window_inds[1]]]
        self.high_window = [self.f[self.high_window_inds[0]], self.f[self.high_window_inds[1]]]
        


    def print_frequency_information(self):
        f = self.f
        print("")
        print("FREQUENCY ARRAY INFORMATION")
        print("----------------------------")
        print(f"Frequency array ranges from {f[0]:.2f} to {f[-1]:.2f} Hz with {len(f)} elements (df = {self.df:.3f} Hz). ")
        print(f"Desired | Actual low-frequency band:   {self.low_window_desired[0]:7.3f} -{self.low_window_desired[1]:7.3f} Hz | {self.low_window[0]:7.3f} -{self.low_window[1]:7.3f} Hz")
        print(f"Desired | Actual high-frequency band:  {self.high_window_desired[0]:7.3f} -{self.high_window_desired[1]:7.3f} Hz | {self.high_window[0]:7.3f} -{self.high_window[1]:7.3f} Hz")

    def print_calibration_information(self):
        ncalib = len(self.metadata_calib['event_name'].unique())
        ncalibrec = len(self.metadata_calib)
        nev = len(self.df_records['event_name'].unique())
        nrec = len(self.df_records)
        print("")
        print("CALIBRATION INFORMATION")
        print("----------------------------")
        print(f"Calibration range:   M {self.calib_mag_range[0]:.2f} to {self.calib_mag_range[1]:.2f}")
        print(f"Calibration events:  {ncalib:,} ({ncalib/nev*100:.2f}%)")
        print(f"Calibration records: {ncalibrec:,} ({ncalibrec/nrec*100:.2f}%)")
        print("")

    def print_metadata_information(self):
        nev = len(self.df_records['event_name'].unique())
        nst = len(self.df_records['channel_name'].unique())
        nrec = len(self.df_records)
        print("")
        print("METADATA INFORMATION")
        print("----------------------------")
        if nev==self.nevents_initial:
            print(f"Events:   {nev:,}")
        else:
            print(f"Events:   {nev:,} of {self.nevents_initial:,} inital ({nev/self.nevents_initial*100:.2f}%)")

        if nst==self.nchannels_initial:
            print(f"Channels: {nst:,}")
        else:
            print(f"Channels: {nst:,} of {self.nchannels_initial:,} inital ({nst/self.nchannels_initial*100:.2f}%)")
        
        if nrec==self.nrecords_initial:
            print(f"Records:  {nrec:,}")
        else:
            print(f"Records:  {nrec:,} of {self.nrecords_initial:,} inital ({nrec/self.nrecords_initial*100:.2f}%)")

    @staticmethod
    def compute_logbeta_static(spectra, low_f_ind, high_f_ind):
        # spectra is an (N x nf) array
        low_band = np.median(np.log10(spectra[:, low_f_ind[0]:low_f_ind[1]+1]), axis=1)
        high_band = np.median(np.log10(spectra[:, high_f_ind[0]:high_f_ind[1]+1]), axis=1)
        logbeta = high_band - low_band
        return logbeta

    # # this function is a work in progress (not working currently)
    # def estimate_kappa0(self, fc_fixed, d_nearby=40, range_min=20, n_min=50):

    #     # CHECK DISTANCE CALCULATIONS
    #     # far stations are still producing kappa0 results. deldist is station-event distance
    #     print("\nKAPPA0 ESTIMATION")
    #     print("----------------------------")
    #     # first, setup grid of possible slopes
    #     nM = 101
    #     M = np.linspace(-0.3, 0.3, nM)

    #     # d_nearby = 40
    #     # range_min = 20
    #     # n_min = 50

    #     # for explanation plots
    #     x_plot = np.linspace(0, 110, 100)

    #     # For each slope: 
    #     # 1) compute residuals of data points with line of slope M[i] and y-intercept 0
    #     # 2) compute mean of residuals (maybe median?)
    #     # 3) store y-intercepts
    #     # 4) compute sum of square

    #     # remove kappa0 if they exist
    #     if 'kappa0' in self.ch_dep: self.ch_dep.remove('kappa0')
    #     if 'kappa0' in self.df_channels.columns: self.df_channels.drop('kappa0', axis=1, inplace=True)
    #     if 'kappa0' in self.metadata_calib.columns: self.metadata_calib.drop('kappa0', axis=1, inplace=True)
    #     if 'kappa0' in self.df_events.columns: self.df_events.drop('kappa0', axis=1, inplace=True)
    #     if 'kappa0' in self.df_records.columns: self.df_records.drop('kappa0', axis=1, inplace=True)

    #     # self.metadata_calib hold all RECORDS of calibration-sized events
    #     df_calib = self.metadata_calib.copy()

    #     nrec = len(df_calib)
    #     # Remove records that are too far
    #     df_calib = df_calib[df_calib['deldist'] <= d_nearby].reset_index(drop=True)
    #     print(f"{nrec-len(df_calib)} of {nrec} records removed for being too far away.")

    #     # df_sta_calib holds the same information, but grouped by station
    #     df_sta_calib = df_calib.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)
    #     nst = len(df_sta_calib)

    #     # remove rows if the range of deldist is less than range_min km
    #     df_sta_calib = df_sta_calib[df_sta_calib['deldist'].apply(max) - df_sta_calib['deldist'].apply(min) > range_min].reset_index(drop=True)
    #     print(f"{nst-len(df_sta_calib)} of {nst} stations removed for having a distance range less than {range_min} km.")
    #     nst = len(df_sta_calib)

    #     # remove rows if there are fewer than 50 points
    #     df_sta_calib = df_sta_calib[df_sta_calib['deldist'].apply(len) > n_min].reset_index(drop=True)
    #     print(f"{nst-len(df_sta_calib)} of {nst} stations removed for having fewer than {n_min} points.")
    #     print(f"{len(df_sta_calib)} stations remaining for kappa0 slope calibration.")

    #     # Do the same, but include events of all distances
    #     df_calib_all = self.metadata_calib.copy()

    #     df_sta_calib_all = df_calib_all.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)

    #     # remove rows if the range of deldist is less than range_min km
    #     df_sta_calib_all = df_sta_calib_all[df_sta_calib_all['deldist'].apply(max) - df_sta_calib_all['deldist'].apply(min) > range_min].reset_index(drop=True)

    #     # remove rows if there are fewer than 50 points
    #     df_sta_calib_all = df_sta_calib_all[df_sta_calib_all['deldist'].apply(len) > n_min].reset_index(drop=True)
    #     print(f"{len(df_sta_calib_all)} stations remaining for kappa0 estimation.")
        

    #     # # only nearby events
    #     # metadata_calib_nearby = self.metadata_calib[self.metadata_calib['deldist'] <= d_nearby].reset_index(drop=True)
    #     # df_sta_calib_nearby = metadata_calib_nearby.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)


    #     # # remove rows if the range of deldist is less than 30 km
    #     # df_sta_calib_nearby = df_sta_calib_nearby[df_sta_calib_nearby['deldist'].apply(max) - df_sta_calib_nearby['deldist'].apply(min) > range_min].reset_index(drop=True)

    #     # # # remove rows if there are fewer than 50 points
    #     # df_sta_calib_nearby = df_sta_calib_nearby[df_sta_calib_nearby['deldist'].apply(len) > n_min].reset_index(drop=True)

        


    #     Y0 = np.zeros((nM, len(df_sta_calib)), dtype=float)
    #     errs = np.zeros(nM)
    #     for i in trange(nM):
    #         for j, row in df_sta_calib.iterrows():

    #             # Store x and y data arrays
    #             deldist = np.array(row['deldist'], dtype=float)
    #             logbeta = np.array(row['logbeta'], dtype=float)
                
    #             # Compute residuals
    #             res = M[i]*deldist - logbeta

    #             # Store y-intercept
    #             Y0[i, j] = np.mean(res)

    #             # Compute shifted residuals (res2 should have zero mean)
    #             res2 = res - Y0[i, j]

    #             # Store the sum of error**2
    #             # if np.isnan(errs[i]): errs[i] = 0
    #             errs[i] += np.sum(res2**2)
    #             # errs[i] += np.sum(np.abs(res2))

    #     imin = np.argmin(errs)

    #     # fit a parabola to nearest +/- 20 points around minimum using np.polyfit
    #     X = M[imin-20:imin+20]
    #     Y = errs[imin-20:imin+20]
    #     p = np.polyfit(X, Y, 2)
    #     mfit = -p[1]/(2*p[0])

    #     print(f"Best-fit slope: {mfit:9.6f}")

    #     # kappa0 computation
    #     brune = 1 / (1 + (self.f/fc_fixed)**2)
    #     logbeta_brune = self.compute_logbeta_static(brune.reshape((1,len(brune))), self.low_window_inds, self.high_window_inds)
    #     print("logbeta_brune = ", logbeta_brune)
    #     f_h = np.mean(self.high_window)
    #     f_l = np.mean(self.low_window)



    #     # recompute y0 using this best-fit slope
    #     y0 = np.zeros(len(df_sta_calib_all), dtype=float)
    #     kappa0 = np.zeros(len(df_sta_calib_all), dtype=float)
    #     nev = np.zeros(len(df_sta_calib_all), dtype=int)
    #     err = 0
    #     for j, row in df_sta_calib_all.iterrows():
    #         # Store x and y data arrays
    #         deldist = np.array(row['deldist'], dtype=float)
    #         logbeta = np.array(row['logbeta'], dtype=float)
    #         nev[j] = len(deldist)
            
    #         # Compute residuals
    #         res = logbeta - mfit*deldist

    #         # Store y-intercept
    #         y0[j] = np.mean(res)

    #         # Compute shifted residuals (res2 should have zero mean)
    #         res2 = res - y0[j]

    #         # Store the sum of error**2
    #         err += np.sum(res2**2)

    #         # debug plots

    #         # plt.figure()
    #         # plt.scatter(deldist, logbeta, c='k', marker='.', s=1)
    #         # plt.plot(x, mfit*x + y0[j], c='r', lw=2)
    #         # plt.axhline(logbeta_brune, c='g', lw=2)
    #         # plt.show()

    #         kappa0[j] = (y0[j] - logbeta_brune) / (-np.pi * np.log10(np.e) * (f_h - f_l))

    #         A0 = np.exp(-np.pi * kappa0[j] * self.f)

    #         # Qex = np.exp()

    #         # example plots
    #         mean_spec = np.mean(np.array(row['s2']), axis=0)

    #         shift = np.mean(brune[self.low_window_inds[0]:self.low_window_inds[1]+1]) / np.mean(mean_spec[self.low_window_inds[0]:self.low_window_inds[1]+1])

    #         if j < 0:
    #         # if kappa0[j] < 0:
    #         # if kappa0[j] > 0.06:
    #         # if kappa0[j] > 0.02 and kappa0[j] < 0.03:
    #             yrange = 7
    #             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4), layout='constrained')
    #             ax1.plot(self.f, np.array(row['s2']).T, c='grey', lw=1, label='Calibration events')
    #             ax1.plot(self.f, mean_spec, c='k', lw=2, label='Mean spectrum')
    #             ax1.plot(self.f, brune / shift, c='r', lw=2, label='Brune')
    #             # ax1.plot(self.f, A0 / shift, c='g', lw=2, label='exp(-pi*kappa0*f)')
    #             ax1.axvline(self.f[1], c='k', lw=2)

    #             # plot low window as rectangle
    #             ax1.axvspan(self.low_window[0], self.low_window[1], color='grey', alpha=0.2)
    #             ax1.axvspan(self.high_window[0], self.high_window[1], color='grey', alpha=0.2)
                
    #             #horizontal line in low band at y=logmean of mean spectrum
    #             ax1.plot(self.low_window, np.ones(2) * 10**np.log10(np.mean(mean_spec[self.low_window_inds[0]:self.low_window_inds[1]+1])), color='k', linestyle='--', lw=2)
    #             ax1.plot(self.low_window, np.ones(2) * 10**np.log10(np.mean(brune[self.low_window_inds[0]:self.low_window_inds[1]+1])/shift), color='r', linestyle='--', lw=2)

    #             # same for high window
    #             ax1.plot(self.high_window, np.ones(2) * 10**np.log10(np.mean(mean_spec[self.high_window_inds[0]:self.high_window_inds[1]+1])), color='k', linestyle='--', lw=2)
    #             ax1.plot(self.high_window, np.ones(2) * 10**np.log10(np.mean(brune[self.high_window_inds[0]:self.high_window_inds[1]+1])/shift), color='r', linestyle='--', lw=2)

    #             ax1.set_xscale('log')
    #             ax1.set_yscale('log')

    #             # title: channel_name | n= 
    #             ax1.set_title(f"{row['channel_name']} | n = {len(deldist)} | kappa0 = {kappa0[j]:5.3e}")
    #             ax1.set_xlabel('Frequency (Hz)')
    #             ax1.set_xlim((self.f[1], 40))
    #             ax1.set_ylim([mean_spec[1]/(10**yrange), mean_spec[1]*10])
    #             handles, labels = ax1.get_legend_handles_labels()
    #             by_label = dict(zip(labels, handles))
    #             ax1.legend(by_label.values(), by_label.keys())

    #             ax2.scatter(deldist, logbeta, c='k', marker='.', s=1)
    #             ax2.plot(x_plot, mfit*x_plot + y0[j], c='r', lw=2)
    #             # ax2.axhline(logbeta_brune, c='g', lw=2)
    #             ax2.axhline(0, c='k', lw=2)

    #             plt.show()
    #     df_sta_calib_all['kappa0'] = kappa0
        
    #     # print(self.df_channels)
    #     self.group_channels()
    #     # self.df_sta = self.df_records.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)
    #     # self.df_sta['kappa0'] = np.nan
        

    #     df_kappa = df_sta_calib_all[['_cid', 'kappa0']]
    #     print(df_kappa)

    #     self.df_channels = pd.merge(self.df_channels, df_kappa, how='left', on='_cid')
    #     if 'kappa0' not in self.ch_dep:
    #         self.ch_dep += ['kappa0']
    #     self.df_sta_calib = df_sta_calib_all
    #     return self



def _get_inds_of_values_in_array(x, values):
    return np.array([np.argmin(np.abs(x - val)) for val in values], dtype=int)

def _get_id_from_column(df, column, id_name):
    # using column name column, get unique values in df[column], 
    # assign each a unique integer, and add that as a new column named 
    # id_name
    ids = df[column].unique()
    id_dict = {ids[i]: i for i in range(len(ids))}
    df[id_name] = df[column].map(id_dict)
    return df, id_dict

def _haversine_km(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1r, lat1r, lon2r, lat2r = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlonr = lon2r - lon1r 
    dlatr = lat2r - lat1r 
    a = np.sin(dlatr/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlonr/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371.0 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
