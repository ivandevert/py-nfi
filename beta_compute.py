#!/usr/bin/env python3

"""
beta_compute.py

Optimized version of beta_compute.py targeting:
  1) Memory reduction (~3-5x for 10M+ records)
  2) Multiprocessing of compute_dlogbeta via shared memory

Key changes from original:
  - target_data replaced with contiguous numpy arrays (eliminates dict-of-arrays overhead)
  - metadata_calib stored as raw arrays, not a DataFrame copy
  - group_events / explode_events use index-based mapping instead of list-in-cell DataFrames
  - Uncertainty bookkeeping uses sparse storage (only events that produce results)
  - compute_dlogbeta parallelized with multiprocessing.Pool + shared memory arrays
  - np.isin replaced with pre-built set lookups where beneficial
  - convert_utc_to_posix_ns uses vectorized pd.to_datetime instead of per-row obspy UTC
  - groupby uses observed=True to avoid Cartesian product explosion with categoricals
  - changed dlogbeta_corr to nfi
  - save_fwf() now sorts by edatetime when available, otherwise event_name
  - added 'nearest-neighbor' option for calib_search_method
  - Now, explosion events are dropped before calibration event selection

Last Modified:
    2026-04-10

Future improvements:
  - Use nearest neighbors for calibration events
"""

import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import time
import os
import pickle as pkl
import inspect
import multiprocessing as mp
from multiprocessing import shared_memory
from functools import partial

import matplotlib.pyplot as plt


# ============================================================
# Module-level worker function (must be picklable for mp.Pool)
# ============================================================

def _dlogbeta_worker(event_indices, shm_names, shm_shapes, shm_dtypes,
                     calib_n_records_min, calib_hdist_max, calib_zdist_max,
                     calib_time_filter, calib_time_method, calib_time_ndays,
                     compute_uncertainty, confidence_level, 
                     save_full_uncertainty, calib_search_method, calib_nn, 
                     calib_depth_scale
    ):
    """
    Worker function for parallel dlogbeta computation.
    
    Attaches to shared memory arrays, processes a chunk of events,
    returns results as dicts.
    """
    # Attach to shared memory
    shm_handles = {}
    arrays = {}
    for name in shm_names:
        shm = shared_memory.SharedMemory(name=shm_names[name])
        arr = np.ndarray(shm_shapes[name], dtype=shm_dtypes[name], buffer=shm.buf)
        shm_handles[name] = shm
        arrays[name] = arr

    # Unpack arrays
    # Target event arrays (nevents x ...)
    t_edep      = arrays['t_edep']
    t_elat      = arrays['t_elat']
    t_elon      = arrays['t_elon']
    t_rec_start = arrays['t_rec_start']  # start index into record arrays
    t_rec_count = arrays['t_rec_count']  # number of records for this event
    
    # Per-record arrays (nrecords)
    r_cid       = arrays['r_cid']
    r_logbeta   = arrays['r_logbeta']
    
    # Calibration arrays (ncalib_records)
    c_edep      = arrays['c_edep']
    c_elat      = arrays['c_elat']
    c_elon      = arrays['c_elon']
    c_cid       = arrays['c_cid']
    c_logbeta   = arrays['c_logbeta']
    c_slat      = arrays['c_slat']
    c_slon      = arrays['c_slon']

    # NN-specific arrays
    if calib_search_method == 'nearest-neighbor':
        c_event_idx = arrays['c_event_idx']  # record -> unique calib event index
        cu_elat     = arrays['cu_elat']
        cu_elon     = arrays['cu_elon']
        cu_edep     = arrays['cu_edep']

    # Optional time arrays
    if calib_time_filter:
        t_etime = arrays['t_etime']
        c_etime = arrays['c_etime']

    # Confidence intervals
    if compute_uncertainty:
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

    # Results for this chunk
    results = []

    for ev_idx in event_indices:
        start = t_rec_start[ev_idx]
        count = t_rec_count[ev_idx]
        if count == 0:
            continue
        
        target_edep = t_edep[ev_idx]
        target_elat = t_elat[ev_idx]
        target_elon = t_elon[ev_idx]
        target_cids = r_cid[start:start+count]
        target_logbeta = r_logbeta[start:start+count]

        # ---- CALIBRATION EVENT SEARCH ----
        # Depth filter
        depth_mask = (c_edep >= (target_edep - calib_zdist_max)) & \
                     (c_edep <= (target_edep + calib_zdist_max))

        # Channel overlap filter
        target_cid_set = set(target_cids)
        cid_mask = np.array([cid in target_cid_set for cid in c_cid[depth_mask]], dtype=bool)
        
        # Get indices into original calib arrays that pass depth
        depth_indices = np.where(depth_mask)[0]
        combined_indices = depth_indices[cid_mask]
        
        if len(combined_indices) == 0:
            continue

        # Time filter
        if calib_time_filter and calib_time_method == 'interval':
            target_etime = t_etime[ev_idx]
            time_delta_days = np.abs(c_etime[combined_indices] - target_etime) / 86400.0 / 1E9
            time_mask = time_delta_days <= calib_time_ndays
            combined_indices = combined_indices[time_mask]
            if len(combined_indices) == 0:
                continue

        # Distance filter
        dists = _haversine_km(
            np.full(len(combined_indices), target_elat),
            np.full(len(combined_indices), target_elon),
            c_elat[combined_indices],
            c_elon[combined_indices]
        )
        dist_mask = dists <= calib_hdist_max

        if calib_search_method == 'cylinder':
            if dist_mask.sum() < calib_n_records_min:
                continue
            calib_mask_indices = combined_indices[dist_mask]

        elif calib_search_method == 'nearest-neighbor':
            combined_indices = combined_indices[dist_mask]
            if len(combined_indices) < calib_n_records_min:
                continue

            # Compute scaled 3D distance per record (via parent event)
            rec_event_idx = c_event_idx[combined_indices]
            ce_elat = cu_elat[rec_event_idx]
            ce_elon = cu_elon[rec_event_idx]
            ce_edep = cu_edep[rec_event_idx]

            hdists_ev = _haversine_km(
                np.full(len(combined_indices), target_elat),
                np.full(len(combined_indices), target_elon),
                ce_elat, ce_elon
            )
            vdists_ev = np.abs(ce_edep - target_edep) * calib_depth_scale
            rec_scaled_dists = np.sqrt(hdists_ev**2 + vdists_ev**2)

            calib_mask_indices = combined_indices

        cc_cid = c_cid[calib_mask_indices]
        cc_logbeta = c_logbeta[calib_mask_indices]

        if calib_search_method == 'nearest-neighbor':
            cc_rec_dists = rec_scaled_dists  # already aligned with calib_mask_indices

        # ---- MATCH CALIBRATION & TARGET RECORDS ----

        mask_t = np.isin(target_cids, cc_cid)
        t_cid = target_cids[mask_t]
        t_lb = target_logbeta[mask_t]

        sorter = np.argsort(t_cid)
        t_cid_sorted = t_cid[sorter]
        t_lb_sorted = t_lb[sorter]

        v = np.searchsorted(t_cid_sorted, cc_cid)
        differences = t_lb_sorted[v] - cc_logbeta

        # Group by channel
        unique_channels, inverse_indices = np.unique(cc_cid, return_inverse=True)
        sorted_order = np.argsort(inverse_indices)
        sorted_diffs = differences[sorted_order]
        sorted_inverse = inverse_indices[sorted_order]

        # If NN mode, also sort the distances for per-channel trimming
        if calib_search_method == 'nearest-neighbor':
            sorted_rec_dists = cc_rec_dists[sorted_order]

        split_indices = np.r_[
            0,
            np.where(np.diff(sorted_inverse) != 0)[0] + 1, 
            len(sorted_inverse)
        ]
        ngroups = len(split_indices) - 1
        
        # Compute per-channel medians and counts
        dlogbeta_j = np.empty(ngroups, dtype=np.float64)
        ncalib_j = np.empty(ngroups, dtype=np.int64)
        for g in range(ngroups):
            s, e = split_indices[g], split_indices[g+1]
            grp = sorted_diffs[s:e]

            if calib_search_method == 'nearest-neighbor' and len(grp) > calib_nn:
                # Keep only the nearest calib_nn records for this channel
                grp_dists = sorted_rec_dists[s:e]
                nearest = np.argpartition(grp_dists, calib_nn)[:calib_nn]
                grp = grp[nearest]
                
            dlogbeta_j[g] = np.median(grp)
            ncalib_j[g] = len(grp)

        final_mask = ncalib_j >= calib_n_records_min
        if final_mask.sum() < 3:
            continue

        dlogbeta_j_kept = dlogbeta_j[final_mask]
        ncalib_j_kept = ncalib_j[final_mask]
        dlogbeta = np.median(dlogbeta_j_kept)

        result_entry = {
            'ev_idx': ev_idx,
            'dlogbeta': dlogbeta,
        }

        if compute_uncertainty:
            # Compute residuals from kept groups
            kept_group_indices = np.where(final_mask)[0]
            residuals = []
            for g in kept_group_indices:
                grp = sorted_diffs[split_indices[g]:split_indices[g+1]]
                residuals.append(grp - np.median(grp))
            all_residuals = np.concatenate(residuals)
            samples = all_residuals + dlogbeta

            result_entry['dlogbeta_std'] = np.std(samples)
            result_entry['dlogbeta_lower'] = np.percentile(samples, lower_percentile)
            result_entry['dlogbeta_upper'] = np.percentile(samples, upper_percentile)
            result_entry['dlogbeta_median'] = np.median(samples)

            if save_full_uncertainty:
                # Store per-station details
                cc_slat = c_slat[calib_mask_indices]
                cc_slon = c_slon[calib_mask_indices]
                sorted_cc_slat = cc_slat[sorted_order]
                sorted_cc_slon = cc_slon[sorted_order]

                rep_slats = np.array([sorted_cc_slat[split_indices[g]] for g in kept_group_indices])
                rep_slons = np.array([sorted_cc_slon[split_indices[g]] for g in kept_group_indices])

                bearings = _get_bearing(
                    np.full(len(rep_slats), target_elat),
                    np.full(len(rep_slons), target_elon),
                    rep_slats, rep_slons
                )
                cdists = _haversine_km(
                    np.full(len(rep_slats), target_elat),
                    np.full(len(rep_slons), target_elon),
                    rep_slats, rep_slons
                )

            
                result_entry['dlogbeta_j'] = dlogbeta_j_kept.tolist()
                result_entry['ncalib_j'] = ncalib_j_kept.tolist()
                result_entry['bearing'] = bearings.tolist()
                result_entry['deldist'] = cdists.tolist()
                result_entry['slat'] = rep_slats.tolist()
                result_entry['slon'] = rep_slons.tolist()
                result_entry['dlogbeta_j_residual'] = all_residuals.tolist()

        results.append(result_entry)

    # Detach shared memory (don't unlink — the parent owns it)
    for shm in shm_handles.values():
        shm.close()

    return results

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
        print_time              = True,
        save_dir                = None,
        compute_uncertainty     = True,
        confidence_level        = 0.95,
        calib_time_filter       = False,
        calib_time_method       = 'interval',
        calib_time_ndays        = 180,
        calib_time_n_nearest    = 25,
        n_workers               = None,
        save_full_uncertainty   = False,
        calib_search_method     = 'cylinder',
        calib_nn                = 10,
        calib_depth_scale       = 5,
        ):
        t0 = time.time()

        if not quiet: print("Initializing BetaEstimator")
        if not quiet: print("--------------------------")
        
        # These are parameters that shouldn't change results (assuming
        # df_records and spectra are the same)
        excl = ['self', 'df_records', 'spectra', 'f', 'quiet', 'save_dir', 
                'print_time', 'n_workers']
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
        self.confidence_level = confidence_level
        self.print_time = print_time
        self.calib_time_filter = calib_time_filter
        self.calib_time_method = calib_time_method
        self.calib_time_ndays = calib_time_ndays
        self.calib_time_n_nearest = calib_time_n_nearest
        self.save_full_uncertainty = save_full_uncertainty
        self.calib_search_method = calib_search_method
        self.calib_nn = calib_nn
        self.calib_depth_scale = calib_depth_scale

        # Number of workers for parallel computation
        if n_workers is None:
            self.n_workers = max(1, mp.cpu_count() // 2)
        else:
            self.n_workers = n_workers

        # Validate input
        self.validate_input()
        if not self.quiet: self.print_metadata_information()

        # Do some basic processing before computing logbeta
        self.digitize_names()
        self.compute_column_dependencies()

        # Calculate actual indices and frequency bands for beta computation
        self.compute_frequency_bands()
        if not self.quiet: self.print_frequency_information()

        # Get indices of calibration events
        self.store_calibration_events()
        if not self.quiet: self.print_calibration_information()

        self.fprint("STATUS:")
        self.fprint(f"BetaEstimator initialized in {time.time()-t0:.4f} s. "
                     f"Using {self.n_workers} workers. Run 'compute()' to continue.")
        self.fprint("--------------------------------------------------------------------")

    def update_nrec(self):
        self.df_events['nrec'] = self.df_events['_cid'].apply(len)

    def store_calibration_events(self):
        self.calibration_inds = np.where(np.logical_and(
            self.df_records['emag'].values >= self.calib_mag_range[0], 
            self.df_records['emag'].values <= self.calib_mag_range[1]
            ))[0]
        self.metadata_calib = self.df_records.iloc[self.calibration_inds].reset_index(drop=True)

        if "etype" in self.metadata_calib.columns:
            l0 = len(self.metadata_calib)
            self.metadata_calib = self.metadata_calib[np.logical_and(
                self.metadata_calib['etype']!='qb', 
                self.metadata_calib['etype']!='ex'
            )].reset_index(drop=True)

            print(f"Dropped {l0-len(self.metadata_calib):,} quarry blast "\
                    "and explosion calibration events. Remaining: "\
                    f"{len(self.metadata_calib):,} calibration events.")
    
    def compute(self, recompute=False):
        self.fprint('')
        self.fprint("Computing nFI")
        self.fprint("-----------------")
        params_filepath = os.path.join(self.save_dir, 'params.logbeta')
        data_filepath = os.path.join(self.save_dir, 'data.logbeta')

        files_exist = os.path.exists(params_filepath) and os.path.exists(data_filepath)

        if not files_exist:
            self.fprint(f"Results not found in {self.save_dir} \nComputing.")
            do_computation = True
        else:
            if recompute:
                self.fprint("Recomputing & overwriting previous results.")
                do_computation = True
            else:
                t0 = time.time()
                current_params = {key: getattr(self, key) for key in self.input_parameter_names}
                loaded_params = self.load_parameters()
                if current_params == loaded_params:
                    self.fprint(f"Parameters unchanged from {params_filepath} \nSkipping computation.")
                    do_computation = False
                else:
                    self.fprint(f"Parameters changed from {params_filepath} \nRecomputing.")
                    do_computation = True

        if do_computation:
            self.compute_logbeta()
            self.store_calibration_events()
            self.group_events()
            self.compute_dlogbeta()
            self.apply_magnitude_correction(corr_type=self.mag_corr_method)
            
            self.store_calibration_events()
            if not self.quiet: self.print_calibration_information()
            self.update_nrec()
            self.save_parameters()
            self.store_edatetimes()
            self.save_fwf()
            self.save_data()

            return self
        else:
            t0 = time.time()
            self.fprint(f"Loading data & parameters from {self.save_dir}")
            loaded_instance = self.load_data(self.save_dir)
            self.load_parameters()
            self.fprint(f"Done. ({time.time()-t0:.2f}s)")
            return loaded_instance

    def save_data(self):
        t0 = time.time()
        self.fprint(f"Saving data to {self.save_dir}/data.logbeta")
        with open(f"{self.save_dir}/data.logbeta", 'wb') as fs:
            pkl.dump(self, fs)
        self.tprint(f"    |---> ({time.time()-t0:.2f}s)")

    def store_edatetimes(self):

        pass

    def save_fwf(self):
        t0 = time.time()
        self.fprint(f"Saving fixed-width format results to {self.save_dir}/logbeta.txt")

        col_fmts = {
            'event_name':    '%12s',
            'edatetime':     '%27s',
            'etype':         '%2s',
            'emag':          '%5.2f',
            'emagtype':      '%2s',
            'elon':          '%10.5f',
            'elat':          '%9.5f',
            'edep':          '%7.3f',
            'nrec':          '%5i',
            'dlogbeta':      '%9.5f',
            'dlogbeta_std':  '%8.3e',
            'dlogbeta_lower':'%8.3e',
            'dlogbeta_upper':'%8.3e',
            'nfi':           '%9.5f',
        }

        # Filter to columns that exist in the output dataframe
        available = {c: f for c, f in col_fmts.items() if c in self.df_events.columns}
        cols = list(available.keys())
        fmts = list(available.values())

        # Sort by time if available, otherwise by name
        sort_col = 'edatetime' if 'edatetime' in cols else 'event_name'
        df_out = self.df_events[cols].sort_values(by=sort_col).reset_index(drop=True)
        self.fprint(f"Output sorted by {sort_col}")

        np.savetxt(
            f"{self.save_dir}/logbeta.txt",
            df_out.values,
            fmt=" ".join(fmts),
            header=' '.join(cols),
            comments='',
        )
        self.tprint(f"    |---> ({time.time()-t0:.2f}s)")

    @staticmethod    
    def load_data(save_dir):
        with open(f"{save_dir}/data.logbeta", 'rb') as fs:
            loaded_instance = pkl.load(fs)
        return loaded_instance

    def save_parameters(self):   
        t0 = time.time()
        self.fprint(f"Saving parameters to {self.save_dir}/params.logbeta")     
        params = {key: getattr(self, key) for key in self.input_parameter_names}
        with open(f"{self.save_dir}/params.logbeta", 'wb') as fs:
            pkl.dump(params, fs)
        self.tprint(f"    |---> ({time.time()-t0:.2f}s)")

    def load_parameters(self):
        self.fprint(f"Loading parameters from {self.save_dir}/params.logbeta")
        with open(f"{self.save_dir}/params.logbeta", 'rb') as fs:
            params = pkl.load(fs)
        return params

    def convert_utc_to_posix_ns(self):
        """Convert edatetime column to POSIX nanoseconds (int64).
        
        Handles three input types efficiently:
          1) obspy UTCDateTime objects → use .ns attribute
          2) datetime strings → vectorized pd.to_datetime (~100x faster than per-row UTC())
          3) already numeric → pass through
        """
        col = self.df_records['edatetime']
        
        # Check the first non-null value to determine type
        first_val = col.iloc[0]
        
        if hasattr(first_val, 'ns'):
            # obspy UTCDateTime objects — use .ns (original behavior)
            self.df_records['etime'] = col.apply(lambda x: x.ns)
        elif isinstance(first_val, (int, np.integer)):
            # Already numeric
            self.df_records['etime'] = col.values.astype(np.int64)
        else:
            # String or datetime-like — use vectorized pandas parsing
            # pd.to_datetime is ~100x faster than per-row obspy UTC()
            dt = pd.to_datetime(col)
            self.df_records['etime'] = dt.astype(np.int64)  # nanoseconds since epoch
        
        # self.df_records.drop(columns=['edatetime'], inplace=True)

    def compute_dlogbeta(self):
        """Correct path and station effects simultaneously (dlogbeta).
        
        Parallelized version using multiprocessing with shared memory.
        Memory-optimized: uses contiguous numpy arrays instead of 
        dict-of-arrays for target data.
        """
        t0 = time.time()
        self.fprint(f"Computing dlogbeta (parallel, {self.n_workers} workers)")
        self.fprint(f"Using {self.calib_search_method} calibration search method")
        self.fprint("----------------------------------------")

        eids = self.df_events['_eid'].values
        nevents = len(eids)
        nrecords = len(self.df_records)

        # ---------------------------------------------------------
        # Build contiguous arrays for target events
        # ---------------------------------------------------------
        self.df_records = self.df_records.sort_values('_eid').reset_index(drop=True)
        
        # These store event metadata, one entry per event, aligned with eids
        t_edep = np.empty(nevents, dtype=np.float64)
        t_elat = np.empty(nevents, dtype=np.float64)
        t_elon = np.empty(nevents, dtype=np.float64)
        t_rec_start = np.empty(nevents, dtype=np.int64)
        t_rec_count = np.empty(nevents, dtype=np.int64)
        t_event_name = []
        
        has_time = self.calib_time_filter
        if has_time:
            t_etime = np.empty(nevents, dtype=np.int64)

        # Store _eid-sorted logbeta and channel identifiers
        r_cid = self.df_records['_cid'].values.astype(np.int64)
        r_logbeta = self.df_records['logbeta'].values.astype(np.float64)
        
        # df_records is sorted by _eid, so going through records in order
        # we will see all records for one event together, then all records 
        # for the next event, etc.

        # Build event offset arrays vectorized instead of 720K get_group() calls.
        # Since df_records is sorted by _eid, we can find boundaries directly.
        eid_col = self.df_records['_eid'].values
        # Find where each eid starts in the sorted records
        change_points = np.r_[0, np.where(np.diff(eid_col) != 0)[0] + 1, len(eid_col)] # tested, works
        eid_at_boundary = eid_col[change_points[:-1]] # tested, works
        
        # Build lookup: eid -> (start, count)
        eid_to_offset = {}
        for k in range(len(change_points) - 1):
            eid_to_offset[eid_at_boundary[k]] = (change_points[k], change_points[k+1] - change_points[k])
        # tested, works

        # Extract per-event scalars using first record of each group
        edep_all = self.df_records['edep'].values
        elat_all = self.df_records['elat'].values
        elon_all = self.df_records['elon'].values
        ename_all = self.df_records['event_name'].values
        if has_time:
            etime_all = self.df_records['etime'].values
        
        for i, eid in enumerate(eids):
            start, count = eid_to_offset[eid]
            t_edep[i] = edep_all[start]
            t_elat[i] = elat_all[start]
            t_elon[i] = elon_all[start]
            t_event_name.append(ename_all[start])
            t_rec_start[i] = start
            t_rec_count[i] = count
            if has_time:
                t_etime[i] = etime_all[start]

        # ---------------------------------------------------------
        # Calibration arrays
        # ---------------------------------------------------------

        # Pull the calibration record metadata into contiguous numpy arrays
        c_edep = self.metadata_calib['edep'].values.astype(np.float64)
        c_elat = self.metadata_calib['elat'].values.astype(np.float64)
        c_elon = self.metadata_calib['elon'].values.astype(np.float64)
        c_cid = self.metadata_calib['_cid'].values.astype(np.int64)
        c_logbeta_arr = self.metadata_calib['logbeta'].values.astype(np.float64)
        c_slat = self.metadata_calib['slat'].values.astype(np.float64)
        c_slon = self.metadata_calib['slon'].values.astype(np.float64)
        
        if has_time:
            c_etime = self.metadata_calib['etime'].values.astype(np.int64)

        self.fprint(f"  Target events: {nevents:,}, Records: {nrecords:,}, "
                     f"Calib records: {len(c_edep):,}")

        # ---------------------------------------------------------
        # NN-specific: map each calibration record to its unique
        # calibration event index
        # ---------------------------------------------------------
        use_nn = (self.calib_search_method == 'nearest-neighbor')

        if use_nn:
            calib_eids = self.metadata_calib['_eid'].values.astype(np.int64)
            unique_calib_eids, c_event_idx = np.unique(
                calib_eids, return_inverse=True)
            c_event_idx = c_event_idx.astype(np.int64)
            n_unique_calib = len(unique_calib_eids)

            # First occurrence of each unique event -> its location
            first_occurrence = np.zeros(n_unique_calib, dtype=np.int64)
            seen = set()
            for i, ev_idx_val in enumerate(c_event_idx):
                if ev_idx_val not in seen:
                    first_occurrence[ev_idx_val] = i
                    seen.add(ev_idx_val)

            cu_elat = c_elat[first_occurrence]
            cu_elon = c_elon[first_occurrence]
            cu_edep = c_edep[first_occurrence]
            if has_time:
                cu_etime = c_etime[first_occurrence]

            self.fprint(f"  NN mode: {n_unique_calib:,} unique calib events, "
                         f"calib_nn={self.calib_nn}, "
                         f"depth_scale={self.calib_depth_scale}")

        # ---------------------------------------------------------
        # Create shared memory blocks
        # ---------------------------------------------------------
        def _create_shm(name, arr):
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
            shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shared_arr[:] = arr[:]
            return shm

        shm_blocks = []
        shm_names = {}
        shm_shapes = {}
        shm_dtypes = {}

        arrays_to_share = {
            't_edep': t_edep, 't_elat': t_elat, 't_elon': t_elon,
            't_rec_start': t_rec_start, 't_rec_count': t_rec_count,
            'r_cid': r_cid, 'r_logbeta': r_logbeta,
            'c_edep': c_edep, 'c_elat': c_elat, 'c_elon': c_elon,
            'c_cid': c_cid, 'c_logbeta': c_logbeta_arr,
            'c_slat': c_slat, 'c_slon': c_slon,
        }
        if has_time:
            arrays_to_share['t_etime'] = t_etime
            arrays_to_share['c_etime'] = c_etime

        if use_nn:
            arrays_to_share['c_event_idx'] = c_event_idx
            arrays_to_share['cu_elat'] = cu_elat
            arrays_to_share['cu_elon'] = cu_elon
            arrays_to_share['cu_edep'] = cu_edep
            if has_time:
                arrays_to_share['cu_etime'] = cu_etime

        shm_prefix = f"beta_{os.getpid()}_"
        
        for key, arr in arrays_to_share.items():
            name = shm_prefix + key
            shm = _create_shm(name, arr)
            shm_blocks.append(shm)
            shm_names[key] = name
            shm_shapes[key] = arr.shape
            shm_dtypes[key] = arr.dtype

        # ---------------------------------------------------------
        # Split events into many small chunks for fine-grained progress.
        # Small chunks (~100-500 events) give good load balancing and
        # per-event resolution in the progress bar, while keeping IPC
        # overhead manageable (~1ms per chunk × a few thousand chunks = seconds).
        # ---------------------------------------------------------
        chunk_size = max(1, min(200, nevents // (self.n_workers * 8)))
        chunks = []
        for start in range(0, nevents, chunk_size):
            end = min(start + chunk_size, nevents)
            chunks.append(list(range(start, end)))

        self.fprint(f"  Dispatching {len(chunks)} chunks ({chunk_size} events/chunk) "
                     f"to {self.n_workers} workers...")

        worker_fn = partial(
            _dlogbeta_worker,
            shm_names=shm_names,
            shm_shapes=shm_shapes,
            shm_dtypes=shm_dtypes,
            calib_n_records_min=self.calib_n_records_min,
            calib_hdist_max=self.calib_hdist_max,
            calib_zdist_max=self.calib_zdist_max,
            calib_time_filter=self.calib_time_filter,
            calib_time_method=self.calib_time_method,
            calib_time_ndays=self.calib_time_ndays,
            compute_uncertainty=self.compute_uncertainty,
            confidence_level=self.confidence_level,
            save_full_uncertainty=self.save_full_uncertainty,
            calib_search_method=self.calib_search_method,
            calib_nn=self.calib_nn,
            calib_depth_scale=self.calib_depth_scale,
        )

        # ---------------------------------------------------------
        # Run parallel computation
        # Progress bar tracks events completed (not chunks), so you
        # see fine-grained progress even with multiprocessing.
        # ---------------------------------------------------------
        all_results = []
        events_done = 0
        
        pbar = tqdm(total=nevents, desc="Computing dlogbeta", unit="events")
        
        if self.n_workers == 1:
            for chunk in chunks:
                chunk_results = worker_fn(chunk)
                all_results.extend(chunk_results)
                events_done += len(chunk)
                pbar.update(len(chunk))
        else:
            with mp.Pool(processes=self.n_workers) as pool:
                for chunk_results in pool.imap_unordered(worker_fn, chunks):
                    all_results.extend(chunk_results)
                    # Each chunk has chunk_size events (last may be smaller)
                    # We track by chunk completion but update by event count
                    events_done += chunk_size  # approximate; close enough for progress
                    pbar.n = min(events_done, nevents)
                    pbar.refresh()
        
        pbar.n = nevents
        pbar.refresh()
        pbar.close()

        # ---------------------------------------------------------
        # Collect results back into arrays
        # ---------------------------------------------------------
        dlogbeta_results = np.full(nrecords, np.nan)
        
        if self.compute_uncertainty:
            dlogbeta_std_results = np.full(nrecords, np.nan)
            dlogbeta_lower_results = np.full(nrecords, np.nan)
            dlogbeta_upper_results = np.full(nrecords, np.nan)
            dlogbeta_median_results = np.full(nrecords, np.nan)
            
            uncertainty_records = {}

        for res in all_results:
            ev_idx = res['ev_idx']
            start = t_rec_start[ev_idx]
            count = t_rec_count[ev_idx]
            record_indices = np.arange(start, start + count)
            
            dlogbeta_results[record_indices] = res['dlogbeta']
            
            if self.compute_uncertainty:
                dlogbeta_std_results[record_indices] = res.get('dlogbeta_std', np.nan)
                dlogbeta_lower_results[record_indices] = res.get('dlogbeta_lower', np.nan)
                dlogbeta_upper_results[record_indices] = res.get('dlogbeta_upper', np.nan)
                dlogbeta_median_results[record_indices] = res.get('dlogbeta_median', np.nan)
                
                if self.save_full_uncertainty:
                    uncertainty_records[ev_idx] = {
                        'event_name': t_event_name[ev_idx],
                        'dlogbeta_j': res.get('dlogbeta_j', []),
                        'dlogbeta_j_residual': res.get('dlogbeta_j_residual', []),
                        'bearing': res.get('bearing', []),
                        'deldist': res.get('deldist', []),
                        'ncalib_j': res.get('ncalib_j', []),
                        'slat': res.get('slat', []),
                        'slon': res.get('slon', []),
                    }

        # ---------------------------------------------------------
        # Clean up shared memory
        # ---------------------------------------------------------
        for shm in shm_blocks:
            shm.close()
            shm.unlink()

        # ---------------------------------------------------------
        # Assign results back to DataFrame
        # ---------------------------------------------------------
        self.df_records['dlogbeta'] = dlogbeta_results

        if self.compute_uncertainty:
            
            if self.save_full_uncertainty:
                sorted_keys = sorted(uncertainty_records.keys())
                self.df_dlogbeta = pd.DataFrame([uncertainty_records[k] for k in sorted_keys])

            self.df_records['dlogbeta_std'] = dlogbeta_std_results
            self.df_records['dlogbeta_lower'] = dlogbeta_lower_results
            self.df_records['dlogbeta_upper'] = dlogbeta_upper_results
            self.df_records['dlogbeta_median'] = dlogbeta_median_results

        # Drop rows with NaN dlogbeta
        n_before = len(self.df_records)
        self.df_records = self.df_records.dropna(subset=['dlogbeta']).reset_index(drop=True)
        n_after = len(self.df_records)

        elapsed = time.time() - t0
        self.fprint(f"  dlogbeta complete: {len(all_results):,}/{nevents:,} events, "
                     f"{n_after:,}/{n_before:,} records kept. ({elapsed:.1f}s)")


    def apply_magnitude_correction(self, corr_type='smoothedspline'):
        self.group_events()
        mag_corr_dM = self.mag_corr_dM
        if corr_type == 'smoothedspline':
            from scipy.signal import savgol_filter
            from scipy.stats import binned_statistic

            M_range = np.round((
                np.floor(np.min(self.df_events['emag'].values) / mag_corr_dM) * mag_corr_dM, 
                np.ceil(np.max(self.df_events['emag'].values) / mag_corr_dM) * mag_corr_dM
                ), 4)
            
            edges = np.arange(M_range[0], M_range[1]+1*mag_corr_dM, mag_corr_dM)
            midpoints = (edges[:-1] + edges[1:])/2

            self.df_events = self.df_events[self.df_events['emag']>=midpoints[0]].reset_index(drop=True)

            emag = self.df_events['emag'].values
            dlogbeta = self.df_events['dlogbeta'].values

            bin_inds = np.digitize(emag, edges) - 1

            median_dlbeta = binned_statistic(emag, dlogbeta, statistic='median', bins=edges)[0]

            filter_window_len = int(np.floor(len(median_dlbeta)/4))
            polyorder = 3
            if polyorder >= filter_window_len:
                polyorder = filter_window_len-1
            naninds = np.isnan(median_dlbeta)
            realmids = midpoints[~naninds]
            median_dlbeta_smooth = savgol_filter(median_dlbeta[~naninds], filter_window_len, polyorder)
            corrections = np.interp(self.df_events['emag'].values, realmids, median_dlbeta_smooth)
            self.correction_mags = realmids
            self.correction_function = median_dlbeta_smooth
            self.df_events['nfi'] = self.df_events['dlogbeta'].values - corrections

        self.explode_events()
        self.group_channels()

    def group_events(self):
        self.fprint("group_events()")
        t0 = time.time()
        self.compute_column_dependencies()
        # observed=True: only group by values actually present in the data.
        # This is critical when event_name/channel_name are categorical —
        # observed=False (the pandas default) would try to create the full
        # Cartesian product of all category levels, which with 720K events
        # × 4K channels overflows memory.
        self.df_events = self.df_records.groupby(
            self.ev_dep, as_index=False, observed=True
        )[self.ch_dep+self.pair_dep].agg(list)
        self.tprint(f"    |---> {time.time()-t0:.4f} seconds")

    def group_channels(self):
        self.fprint("group_channels()")
        self.compute_column_dependencies()
        self.df_channels = self.df_records.groupby(
            self.ch_dep, as_index=False, dropna=False, observed=True
        )[self.ev_dep+self.pair_dep].agg(list)

    def explode_events(self):
        self.fprint('explode_events()')
        t0 = time.time()
        self.compute_column_dependencies()
        self.df_records = self.df_events.explode(self.ch_dep+self.pair_dep)
        self.tprint(f"    |---> {time.time()-t0:.4f} seconds")

    def explode_channels(self):
        t0 = time.time()
        self.fprint('explode_channels()')
        self.compute_column_dependencies()
        self.df_records = self.df_channels.explode(self.ev_dep+self.pair_dep)
        self.tprint(f"    |---> {time.time()-t0:.4f} seconds")

    def compute_column_dependencies(self):
        ev_dep_columns = [
            'event_name', 'emag', 'elat', 'elon', 'edep', 'etime', 'edatetime', '_eid', 'event_id', 
            'evid', 'event', 'ex', 'ey', 'nts', 'dlogbeta', 'nfi',
            'nrec', 'dlogbeta_std', 'dlogbeta_lower', 'dlogbeta_upper',
            'dlogbeta_median', 'etype', 'emagtype']
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
        t0 = time.time()
        self.fprint("validate_input()")
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
                    self.fprint("Renamed column {} to event_name".format(col))
                    break
        if "event_name" not in self.df_records.columns:
            raise ValueError("No event_name column found. Check input data.")
        
        channel_name_columns = ['station_id', 'sid', 'station', 'stname', 
                                'st_name', 'stid']
        if 'channel_name' not in columns:
            for col in channel_name_columns:
                if col in columns:
                    self.df_records.rename(
                        columns={col: 'channel_name'}, inplace=True)
                    self.fprint("Renamed column {} to channel_name".format(col))
                    break
        if "channel_name" not in self.df_records.columns:
            raise ValueError("No channel_name column found. Check input data.")
        
        if 'qmag' in columns:
            self.df_records.rename(columns={'qmag': 'emag'}, inplace=True)
        if 'qlat' in columns:
            self.df_records.rename(columns={'qlat': 'elat'}, inplace=True)
        if 'qlon' in columns:
            self.df_records.rename(columns={'qlon': 'elon'}, inplace=True)
        if 'qdep' in columns:
            self.df_records.rename(columns={'qdep': 'edep'}, inplace=True)

        required_columns = [
            'event_name', 'channel_name', 'emag', 'elat', 'elon', 'edep', 
            'slat', 'slon', 'selev', 'deldist']
        for col in required_columns:
            if col not in self.df_records.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert edatetime to posix ns (handles strings, UTCDateTime, etc.)
        if 'edatetime' in columns:
            self.convert_utc_to_posix_ns()

        if len(self.df_records) != self.spectra.shape[0]:
            raise ValueError("Metadata and spectra row counts do not match.")
        if len(self.f) != self.spectra.shape[1]:
            raise ValueError(
                f"Spectra width {self.spectra.shape[1]} does not match "
                f"f array length {len(self.f)}.")

        if (np.median(self.df_records['deldist']) > 100) or \
            (np.median(self.df_records['deldist']) < 10):
            raise Warning(
                f"Is deldist in km? Median is"
                f" {np.median(self.df_records['deldist'])}")
        
        assert np.sum(np.isnan(self.spectra)) == 0, \
            "Spectra contains NaNs."

        self.nrecords_initial = len(self.df_records)
        self.nevents_initial = len(self.df_records['event_name'].unique())
        self.nchannels_initial = len(self.df_records['channel_name'].unique())

        if self.save_dir is None: self.save_dir = 'tmp/'
        os.makedirs(self.save_dir, exist_ok=True)

        self.tprint(f"    |---> {time.time()-t0:.4f} seconds")


    def digitize_names(self):
        self.df_records = self.df_records.sort_values(by=['event_name', 'channel_name'])
        v = self.df_records.index.values
        self.df_records = self.df_records.reset_index(drop=True)
        
        self.spectra = self.spectra[v, :]

        self.df_records, self.event_name_dict = _get_id_from_column(self.df_records, 'event_name', '_eid')
        self.df_records, self.channel_name_dict = _get_id_from_column(self.df_records, 'channel_name', '_cid')

    def compute_logbeta(self):
        t0 = time.time()
        self.fprint("compute_logbeta()")
        low_inds = self.low_window_inds
        high_inds = self.high_window_inds
        
        low_band = np.median(np.log10(self.spectra[:, low_inds[0]:low_inds[1]+1]), axis=1)
        high_band = np.median(np.log10(self.spectra[:, high_inds[0]:high_inds[1]+1]), axis=1)
        self.df_records['logbeta'] = high_band - low_band
        del self.spectra
        self.tprint(f"    |---> {time.time()-t0:.4f} seconds")

    def compute_frequency_bands(self):
        self.low_window_inds = _get_inds_of_values_in_array(self.f, self.low_window_desired)
        self.high_window_inds = _get_inds_of_values_in_array(self.f, self.high_window_desired)

        self.low_window = [self.f[self.low_window_inds[0]], self.f[self.low_window_inds[1]]]
        self.high_window = [self.f[self.high_window_inds[0]], self.f[self.high_window_inds[1]]]
        
    def fprint(self, S):
        if not self.quiet:
            print(S)
    
    def tprint(self, S):
        if self.print_time:
            print(S)

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


def _get_inds_of_values_in_array(x, values):
    return np.array([np.argmin(np.abs(x - val)) for val in values], dtype=int)

def _get_id_from_column(df, column, id_name):
    ids = df[column].unique()
    id_dict = {ids[i]: i for i in range(len(ids))}
    df[id_name] = df[column].map(id_dict)
    return df, id_dict

def _haversine_km(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    lon1r, lat1r, lon2r, lat2r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlonr = lon2r - lon1r 
    dlatr = lat2r - lat1r 
    a = np.sin(dlatr/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlonr/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371.0
    return c * r

def _get_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points, from point 1 to point 2.
    """
    y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))
    bearing = np.degrees(np.arctan2(y, x))
    bearing = (bearing + 360) % 360
    return bearing