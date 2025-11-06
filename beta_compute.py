#!/usr/bin/env python3


import numpy as np
import pandas as pd
from tqdm import trange
import utm
import time
import os
import pickle as pkl
import inspect

import matplotlib.pyplot as plt
# instead of calib_hdist and zdist max, maybe use ncalib to find the ncalib 
# nearest events? could improve spatial resolution but need to test


class BetaEstimator:
    def __init__(self, df_records, spectra, f,
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
        confidence_level        = 0.95
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
        
        # Validate input
        self.validate_input()
        if not self.quiet: self.print_metadata_information()

        # Do some basic processing before computing logbeta
        self.digitize_names()
        # self.compute_eastings_northings()
        self.compute_column_dependencies()

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
        

    def compute(self, recompute=False):
        params_filepath = f"{self.save_dir}/params.logbeta"
        data_filepath = f"{self.save_dir}/data.logbeta"

        files_exist = os.path.exists(params_filepath) and os.path.exists(data_filepath)

        # If the files don't exist, compute logbeta no matter what        
        if not files_exist:
            print("Output files not found. Computing.")
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
                    print("Same parameters. Skipping computation.")
                    do_computation = False
                else:
                    print("Parameters changed. Recomputing.")
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
            loaded_instance = self.load_data(self.save_dir)
            self.load_parameters()
            # self.save_fwf()
            return loaded_instance


    # this function is a work in progress (not working currently)
    def estimate_kappa0(self, fc_fixed, d_nearby=40, range_min=20, n_min=50):

        # CHECK DISTANCE CALCULATIONS
        # far stations are still producing kappa0 results. deldist is station-event distance
        print("\nKAPPA0 ESTIMATION")
        print("----------------------------")
        # first, setup grid of possible slopes
        nM = 101
        M = np.linspace(-0.3, 0.3, nM)

        # d_nearby = 40
        # range_min = 20
        # n_min = 50

        # for explanation plots
        x_plot = np.linspace(0, 110, 100)

        # For each slope: 
        # 1) compute residuals of data points with line of slope M[i] and y-intercept 0
        # 2) compute mean of residuals (maybe median?)
        # 3) store y-intercepts
        # 4) compute sum of square

        # remove kappa0 if they exist
        if 'kappa0' in self.ch_dep: self.ch_dep.remove('kappa0')
        if 'kappa0' in self.df_channels.columns: self.df_channels.drop('kappa0', axis=1, inplace=True)
        if 'kappa0' in self.metadata_calib.columns: self.metadata_calib.drop('kappa0', axis=1, inplace=True)
        if 'kappa0' in self.df_events.columns: self.df_events.drop('kappa0', axis=1, inplace=True)
        if 'kappa0' in self.df_records.columns: self.df_records.drop('kappa0', axis=1, inplace=True)

        # self.metadata_calib hold all RECORDS of calibration-sized events
        df_calib = self.metadata_calib.copy()

        nrec = len(df_calib)
        # Remove records that are too far
        df_calib = df_calib[df_calib['deldist'] <= d_nearby].reset_index(drop=True)
        print(f"{nrec-len(df_calib)} of {nrec} records removed for being too far away.")

        # df_sta_calib holds the same information, but grouped by station
        df_sta_calib = df_calib.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)
        nst = len(df_sta_calib)

        # remove rows if the range of deldist is less than range_min km
        df_sta_calib = df_sta_calib[df_sta_calib['deldist'].apply(max) - df_sta_calib['deldist'].apply(min) > range_min].reset_index(drop=True)
        print(f"{nst-len(df_sta_calib)} of {nst} stations removed for having a distance range less than {range_min} km.")
        nst = len(df_sta_calib)

        # remove rows if there are fewer than 50 points
        df_sta_calib = df_sta_calib[df_sta_calib['deldist'].apply(len) > n_min].reset_index(drop=True)
        print(f"{nst-len(df_sta_calib)} of {nst} stations removed for having fewer than {n_min} points.")
        print(f"{len(df_sta_calib)} stations remaining for kappa0 slope calibration.")

        # Do the same, but include events of all distances
        df_calib_all = self.metadata_calib.copy()

        df_sta_calib_all = df_calib_all.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)

        # remove rows if the range of deldist is less than range_min km
        df_sta_calib_all = df_sta_calib_all[df_sta_calib_all['deldist'].apply(max) - df_sta_calib_all['deldist'].apply(min) > range_min].reset_index(drop=True)

        # remove rows if there are fewer than 50 points
        df_sta_calib_all = df_sta_calib_all[df_sta_calib_all['deldist'].apply(len) > n_min].reset_index(drop=True)
        print(f"{len(df_sta_calib_all)} stations remaining for kappa0 estimation.")
        

        # # only nearby events
        # metadata_calib_nearby = self.metadata_calib[self.metadata_calib['deldist'] <= d_nearby].reset_index(drop=True)
        # df_sta_calib_nearby = metadata_calib_nearby.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)


        # # remove rows if the range of deldist is less than 30 km
        # df_sta_calib_nearby = df_sta_calib_nearby[df_sta_calib_nearby['deldist'].apply(max) - df_sta_calib_nearby['deldist'].apply(min) > range_min].reset_index(drop=True)

        # # # remove rows if there are fewer than 50 points
        # df_sta_calib_nearby = df_sta_calib_nearby[df_sta_calib_nearby['deldist'].apply(len) > n_min].reset_index(drop=True)

        


        Y0 = np.zeros((nM, len(df_sta_calib)), dtype=float)
        errs = np.zeros(nM)
        for i in trange(nM):
            for j, row in df_sta_calib.iterrows():

                # Store x and y data arrays
                deldist = np.array(row['deldist'], dtype=float)
                logbeta = np.array(row['logbeta'], dtype=float)
                
                # Compute residuals
                res = M[i]*deldist - logbeta

                # Store y-intercept
                Y0[i, j] = np.mean(res)

                # Compute shifted residuals (res2 should have zero mean)
                res2 = res - Y0[i, j]

                # Store the sum of error**2
                # if np.isnan(errs[i]): errs[i] = 0
                errs[i] += np.sum(res2**2)
                # errs[i] += np.sum(np.abs(res2))

        imin = np.argmin(errs)

        # fit a parabola to nearest +/- 20 points around minimum using np.polyfit
        X = M[imin-20:imin+20]
        Y = errs[imin-20:imin+20]
        p = np.polyfit(X, Y, 2)
        mfit = -p[1]/(2*p[0])

        print(f"Best-fit slope: {mfit:9.6f}")

        # kappa0 computation
        brune = 1 / (1 + (self.f/fc_fixed)**2)
        logbeta_brune = self.compute_logbeta_static(brune.reshape((1,len(brune))), self.low_window_inds, self.high_window_inds)
        print("logbeta_brune = ", logbeta_brune)
        f_h = np.mean(self.high_window)
        f_l = np.mean(self.low_window)



        # recompute y0 using this best-fit slope
        y0 = np.zeros(len(df_sta_calib_all), dtype=float)
        kappa0 = np.zeros(len(df_sta_calib_all), dtype=float)
        nev = np.zeros(len(df_sta_calib_all), dtype=int)
        err = 0
        for j, row in df_sta_calib_all.iterrows():
            # Store x and y data arrays
            deldist = np.array(row['deldist'], dtype=float)
            logbeta = np.array(row['logbeta'], dtype=float)
            nev[j] = len(deldist)
            
            # Compute residuals
            res = logbeta - mfit*deldist

            # Store y-intercept
            y0[j] = np.mean(res)

            # Compute shifted residuals (res2 should have zero mean)
            res2 = res - y0[j]

            # Store the sum of error**2
            err += np.sum(res2**2)

            # debug plots

            # plt.figure()
            # plt.scatter(deldist, logbeta, c='k', marker='.', s=1)
            # plt.plot(x, mfit*x + y0[j], c='r', lw=2)
            # plt.axhline(logbeta_brune, c='g', lw=2)
            # plt.show()

            kappa0[j] = (y0[j] - logbeta_brune) / (-np.pi * np.log10(np.e) * (f_h - f_l))

            A0 = np.exp(-np.pi * kappa0[j] * self.f)

            # Qex = np.exp()

            # example plots
            mean_spec = np.mean(np.array(row['s2']), axis=0)

            shift = np.mean(brune[self.low_window_inds[0]:self.low_window_inds[1]+1]) / np.mean(mean_spec[self.low_window_inds[0]:self.low_window_inds[1]+1])

            if j < 0:
            # if kappa0[j] < 0:
            # if kappa0[j] > 0.06:
            # if kappa0[j] > 0.02 and kappa0[j] < 0.03:
                yrange = 7
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4), layout='constrained')
                ax1.plot(self.f, np.array(row['s2']).T, c='grey', lw=1, label='Calibration events')
                ax1.plot(self.f, mean_spec, c='k', lw=2, label='Mean spectrum')
                ax1.plot(self.f, brune / shift, c='r', lw=2, label='Brune')
                # ax1.plot(self.f, A0 / shift, c='g', lw=2, label='exp(-pi*kappa0*f)')
                ax1.axvline(self.f[1], c='k', lw=2)

                # plot low window as rectangle
                ax1.axvspan(self.low_window[0], self.low_window[1], color='grey', alpha=0.2)
                ax1.axvspan(self.high_window[0], self.high_window[1], color='grey', alpha=0.2)
                
                #horizontal line in low band at y=logmean of mean spectrum
                ax1.plot(self.low_window, np.ones(2) * 10**np.log10(np.mean(mean_spec[self.low_window_inds[0]:self.low_window_inds[1]+1])), color='k', linestyle='--', lw=2)
                ax1.plot(self.low_window, np.ones(2) * 10**np.log10(np.mean(brune[self.low_window_inds[0]:self.low_window_inds[1]+1])/shift), color='r', linestyle='--', lw=2)

                # same for high window
                ax1.plot(self.high_window, np.ones(2) * 10**np.log10(np.mean(mean_spec[self.high_window_inds[0]:self.high_window_inds[1]+1])), color='k', linestyle='--', lw=2)
                ax1.plot(self.high_window, np.ones(2) * 10**np.log10(np.mean(brune[self.high_window_inds[0]:self.high_window_inds[1]+1])/shift), color='r', linestyle='--', lw=2)

                ax1.set_xscale('log')
                ax1.set_yscale('log')

                # title: channel_name | n= 
                ax1.set_title(f"{row['channel_name']} | n = {len(deldist)} | kappa0 = {kappa0[j]:5.3e}")
                ax1.set_xlabel('Frequency (Hz)')
                ax1.set_xlim((self.f[1], 40))
                ax1.set_ylim([mean_spec[1]/(10**yrange), mean_spec[1]*10])
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())

                ax2.scatter(deldist, logbeta, c='k', marker='.', s=1)
                ax2.plot(x_plot, mfit*x_plot + y0[j], c='r', lw=2)
                # ax2.axhline(logbeta_brune, c='g', lw=2)
                ax2.axhline(0, c='k', lw=2)

                plt.show()
        df_sta_calib_all['kappa0'] = kappa0
        
        # print(self.df_channels)
        self.group_channels()
        # self.df_sta = self.df_records.groupby(self.ch_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)
        # self.df_sta['kappa0'] = np.nan
        

        df_kappa = df_sta_calib_all[['_cid', 'kappa0']]
        print(df_kappa)

        self.df_channels = pd.merge(self.df_channels, df_kappa, how='left', on='_cid')
        if 'kappa0' not in self.ch_dep:
            self.ch_dep += ['kappa0']
        self.df_sta_calib = df_sta_calib_all
        return self


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

    def compute_dlogbeta(self):

        # Correct path and station effects simultaneously (dlogbeta)
        eids = np.unique(self.df_records['_eid'].values)
        nevents = len(eids)

        # Store NaNs, since we are looping and some might not be computed
        self.df_records['dlogbeta'] = np.nan

        if self.compute_uncertainty:
            self.df_records['dlogbeta_std'] = np.nan
            self.df_records['dlogbeta_lower'] = np.nan
            self.df_records['dlogbeta_upper'] = np.nan
            self.df_records['dlogbeta_median'] = np.nan
            self.pair_dep.extend([
                'dlogbeta_std', 
                'dlogbeta_lower', 
                'dlogbeta_upper', 
                'dlogbeta_median'])

            # set confidence intervals for uncertainty
            alpha = 1 - self.confidence_level
            lower_percentile = 100 * (alpha / 2)
            upper_percentile = 100 * (1 - alpha / 2)

        for i in trange(nevents, desc="Computing corrected logbeta"):
            # t0 = time.time()
            eid = eids[i]

            # Store all entries of target event in md_t
            eid_inds = self.df_records['_eid'] == eid
            md_t = self.df_records[eid_inds]
            md_t = md_t.sort_values(by='_cid')

            # Store values we are about to use
            edep = md_t['edep'].values[0]
            elat = md_t['elat'].values[0]
            elon = md_t['elon'].values[0]

            # Make a copy of the calibration event records DataFrame
            md_c = self.metadata_calib.copy()

            # Filter out calibration event records that:
            #   1) are too shallow or too deep
            #   2) don't share channels with the target event
            filt1 = np.all([
                md_c['edep']>=edep-self.calib_zdist_max, 
                md_c['edep']<=edep+self.calib_zdist_max,
                np.isin(md_c['_cid'].values, md_t['_cid'])
                ], axis=0)
            md_c = md_c[filt1].reset_index(drop=True)

            n_md_c = len(md_c)

            # Now, md_c contains records of calibration events in the correct 
            # depth range and with the same stations as the target event

            # Compute distances between target event (elat, elon) and each
            # calibration event (mc_c['elat'], mc_c['elon'])
            dists = _haversine_km(
                np.full(n_md_c, elat), 
                np.full(n_md_c, elon), 
                md_c['elat'].values, 
                md_c['elon'].values)
            md_c = md_c[dists <= self.calib_hdist_max].reset_index(drop=True)

            # # simplify md_c by removing unnecessary columns
            # md_c = md_c[['channel_name','_eid', '_cid', 'logbeta']]

            calib_n_records = len(md_c)

            
            # only compute dlogbeta if there are enough remaining calibration events
            if calib_n_records >= self.calib_n_records_min:

                # t_pre_test = time.time() - t0

                # Store channels that record target event
                # t_cid_all = np.unique(md_t['_cid'].values)
                t_cid_all = md_t['_cid'].values
                # Store channels that record calibration events (some repeated)
                c_cid_all = md_c['_cid'].values

                # Get indices of calibration events that share a channel with target event
                c_ind = np.where(np.isin(c_cid_all, t_cid_all))[0]
                
                # now, all entries in md_c can be used in computing dlogbeta
                # pandas method (slower)
                # md_c = md_c.iloc[c_ind].reset_index(drop=True)
                # c_cid = md_c['_cid'].values
                
                # numpy method
                c_cid = c_cid_all[c_ind]

                # filter out entries in md_t that don't have a channel in md_c
                t_ind = np.where(np.isin(t_cid_all, c_cid))[0]
                
                # # pandas
                # md_t = md_t.iloc[t_ind].reset_index(drop=True)
                # t_cid = md_t['_cid'].values
                
                # numpy
                t_cid = t_cid_all[t_ind]
                
                # assert len(t_cid) == len(np.unique(t_cid))

                # Match order
                v = np.searchsorted(t_cid, c_cid)
                
                # # using pandas - slower?
                # t_logbeta = md_t['logbeta'].values[v]
                # c_logbeta = md_c['logbeta'].values

                # using numpy
                t_logbeta = md_t['logbeta'].values[t_ind][v]
                c_logbeta = md_c['logbeta'].values[c_ind]

                differences = t_logbeta - c_logbeta
                
                self.df_records.loc[eid_inds, 'dlogbeta'] = np.median(differences)

                if self.compute_uncertainty:
                    bootstrap_estimates = np.zeros(self.n_bootstrap)
                    n_samples = len(differences)
                    for b in range(self.n_bootstrap):
                        resample_indices = np.random.choice(
                            n_samples, size=n_samples, replace=True)
                        resampled_differences = differences[resample_indices]
                        bootstrap_estimates[b] = np.median(resampled_differences)
                    
                    self.df_records.loc[eid_inds, 'dlogbeta_std'] = np.std(bootstrap_estimates)
                    self.df_records.loc[eid_inds, 'dlogbeta_lower'] = np.percentile(
                        bootstrap_estimates, lower_percentile)
                    self.df_records.loc[eid_inds, 'dlogbeta_upper'] = np.percentile(
                        bootstrap_estimates, upper_percentile)
                    self.df_records.loc[eid_inds, 'dlogbeta_median'] = np.median(bootstrap_estimates)
                    
                    # plt.figure(figsize=(10,5))
                    # plt.hist(bootstrap_estimates, bins=np.linspace(-1.5, 1.5, 500))
                    # plt.xlabel('dlogbeta')
                    # plt.ylabel('Count')
                    # plt.title(f'dlogbeta for event {eid} std = {np.std(bootstrap_estimates)}')
                    # plt.show()
                    # if i >= 20: raise ValueError()


        # drop rows with NaN dlogbeta
        self.df_records = self.df_records.dropna(subset=['dlogbeta']).reset_index(drop=True)

    def apply_magnitude_correction(self, corr_type='smoothedspline'):
        self.group_events()
        mag_corr_dM = self.mag_corr_dM
        if corr_type == 'smoothedspline':
            from scipy.signal import savgol_filter

            # Automatically determine the range of magnitude bins
            M_range = np.round((
                np.floor(np.min(self.df_events['emag'].values) / mag_corr_dM) * mag_corr_dM, 
                np.ceil(np.max(self.df_events['emag'].values) / mag_corr_dM) * mag_corr_dM
                ), 4) # fixes rounding errors
            
            # Define the edges and midpoints of the magnitude bins
            edges = np.arange(M_range[0], M_range[1]+1*mag_corr_dM, mag_corr_dM)
            midpoints = (edges[:-1] + edges[1:])/2

            # drop events with emag lower than the first bin (interpolation issue?)
            self.df_events = self.df_events[self.df_events['emag']>=midpoints[0]].reset_index(drop=True)

            # Assign each target event to a magnitude bin
            bin_inds = np.digitize(self.df_events['emag'].values, edges) - 1

            # Compute the median dlogbeta for each magnitude bin
            median_dlbeta = np.empty(len(edges)-1) * np.nan
            # print('computing median_dlbeta')
            for i in range(len(median_dlbeta)):
                median_dlbeta[i] = np.median(self.df_events['dlogbeta'].values[bin_inds==i])

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
        
        self.pair_dep.append('dlogbeta_corr')
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
            fmt=fmt
        )

    def group_events(self):
        print("group_events()")
        # group self.df_records by event dependent fields
        self.compute_column_dependencies()
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
            'event_name', 'emag', 'elat', 'elon', 'edep', '_eid', 'event_id', 
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
        if "event_name" not in columns:
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
        if "channel_name" not in columns:
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
        assert np.sum(np.isnull(self.spectra)) == 0, \
            "Spectra contains NaNs."
        
        # Spectra should not be logarithmic:
        if np.median(
            np.max(p_spectra, axis=1) / np.min(p_spectra, axis=1)
            ) < 10:
            raise Warning(
                "Spectra should be linear but appear to be logarithmic."
                )

        # Store some initial counts
        self.nrecords_initial = len(self.df_records)
        self.nevents_initial = len(self.df_records['event_name'].unique())
        self.nchannels_initial = len(self.df_records['channel_name'].unique())

        if self.save_dir is None: self.save_dir = 'tmp/'
        os.makedirs(self.save_dir, exist_ok=True)


    def digitize_names(self):
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
