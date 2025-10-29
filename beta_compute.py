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
    def __init__(self, metadata_df, spectra, f,
        low_window_desired     = (1.0, 5.0),
        high_window_desired    = (15.0, 22.0),
        calib_mag_range     = (1.4, 1.6),
        calib_hdist_max     = 10.0,
        calib_zdist_max     = 2.0,
        n_calib_min         = 10,
        mag_corr_method     = 'smoothedspline',
        mag_corr_dM         = 0.2,
        quiet               = False,
        save_dir            = None,
        compute_uncertainty = True,
        n_bootstrap         = 1000,
        confidence_level    = 0.95
        ):
        if not quiet:
            print("Initializing BetaEstimator")
            print("--------------------------")

        excl = ['self', 'metadata_df', 'spectra', 'f']
        sig = inspect.signature(self.__init__)
        self.input_parameter_names = [k for k in sig.parameters if k not in excl]
        
        # Store parameters as attributes
        self.metadata_df = metadata_df.copy()
        self.spectra = spectra
        self.f = f
        self.df = f[1] - f[0]
        self.low_window_desired = low_window_desired
        self.high_window_desired = high_window_desired
        self.calib_mag_range = calib_mag_range
        self.calib_hdist_max = calib_hdist_max
        self.calib_zdist_max = calib_zdist_max
        self.n_calib_min = n_calib_min
        self.mag_corr_method = mag_corr_method
        self.mag_corr_dM = mag_corr_dM
        self.quiet = quiet
        self.save_dir = save_dir
        self.compute_uncertainty = compute_uncertainty
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
        # Validate input
        self.validate_input()

        # Do some basic processing before computing logbeta
        self.digitize_names()
        self.compute_eastings_northings()
        self.compute_column_dependencies()

        # Calculate actual indices and frequency bands for beta computation
        self.compute_frequency_bands()

        # Get indices of calibration events
        self.store_calibration_events()

        if not self.quiet:
            print("STATUS:")
            print("Data loaded successfully. Run 'compute()' to continue.")
            print("------------------------------------------------------")

        # self.save_parameters()

    def update_nrec(self):
        self.metadata_ev['nrec'] = self.metadata_ev['_sid'].apply(len)

    def store_calibration_events(self):
        self.calibration_inds = np.where(np.logical_and(
            self.metadata_df['emag'].values >= self.calib_mag_range[0], 
            self.metadata_df['emag'].values <= self.calib_mag_range[1]
            ))[0]
        self.metadata_calib = self.metadata_df.iloc[self.calibration_inds].reset_index(drop=True)
        if not self.quiet: self.print_calibration_information()

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
            # if 'kappa0' not in self.metadata_df.columns:
            #     self.metadata_df['kappa0'] = -999.0
            self.compute_dlogbeta()
            self.apply_magnitude_correction(corr_type=self.mag_corr_method)
            self.store_calibration_events()
            self.update_nrec()
            # self.save_results(filename='data/beta_compute_out_test.txt')
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
        if 'kappa0' in self.st_dep: self.st_dep.remove('kappa0')
        if 'kappa0' in self.metadata_st.columns: self.metadata_st.drop('kappa0', axis=1, inplace=True)
        if 'kappa0' in self.metadata_calib.columns: self.metadata_calib.drop('kappa0', axis=1, inplace=True)
        if 'kappa0' in self.metadata_ev.columns: self.metadata_ev.drop('kappa0', axis=1, inplace=True)
        if 'kappa0' in self.metadata_df.columns: self.metadata_df.drop('kappa0', axis=1, inplace=True)

        # self.metadata_calib hold all RECORDS of calibration-sized events
        df_calib = self.metadata_calib.copy()

        nrec = len(df_calib)
        # Remove records that are too far
        df_calib = df_calib[df_calib['deldist'] <= d_nearby].reset_index(drop=True)
        print(f"{nrec-len(df_calib)} of {nrec} records removed for being too far away.")

        # df_sta_calib holds the same information, but grouped by station
        df_sta_calib = df_calib.groupby(self.st_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)
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

        df_sta_calib_all = df_calib_all.groupby(self.st_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)

        # remove rows if the range of deldist is less than range_min km
        df_sta_calib_all = df_sta_calib_all[df_sta_calib_all['deldist'].apply(max) - df_sta_calib_all['deldist'].apply(min) > range_min].reset_index(drop=True)

        # remove rows if there are fewer than 50 points
        df_sta_calib_all = df_sta_calib_all[df_sta_calib_all['deldist'].apply(len) > n_min].reset_index(drop=True)
        print(f"{len(df_sta_calib_all)} stations remaining for kappa0 estimation.")
        

        # # only nearby events
        # metadata_calib_nearby = self.metadata_calib[self.metadata_calib['deldist'] <= d_nearby].reset_index(drop=True)
        # df_sta_calib_nearby = metadata_calib_nearby.groupby(self.st_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)


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

                # title: station_name | n= 
                ax1.set_title(f"{row['station_name']} | n = {len(deldist)} | kappa0 = {kappa0[j]:5.3e}")
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
        
        # print(self.metadata_st)
        self.group_stations()
        # self.df_sta = self.metadata_df.groupby(self.st_dep, as_index=False)[self.ev_dep+self.pair_dep].agg(list)
        # self.df_sta['kappa0'] = np.nan
        

        df_kappa = df_sta_calib_all[['_sid', 'kappa0']]
        print(df_kappa)

        self.metadata_st = pd.merge(self.metadata_st, df_kappa, how='left', on='_sid')
        if 'kappa0' not in self.st_dep:
            self.st_dep += ['kappa0']
        self.df_sta_calib = df_sta_calib_all
        return self


    def save_data(self):
        with open(f"{self.save_dir}/data.logbeta", 'wb') as fs:
            pkl.dump(self, fs)
    
    def save_fwf(self):
        cols = ['event_name', 'emag', 'elon', 'elat', 'edep', 'nrec', 'dlogbeta', 'dlogbeta_corr']
        fmt = '%8i %5.2f %15.11f %15.11f %7.3f %5i %15.11f %15.11f'
        np.savetxt(f"{self.save_dir}/logbeta.txt", self.metadata_ev[cols].values, fmt=fmt)

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
        eids = np.unique(self.metadata_df['_eid'].values)
        nevents = len(eids)

        # Store NaNs, since we are looping and some might not be computed
        self.metadata_df['dlogbeta'] = np.nan

        if self.compute_uncertainty:
            self.metadata_df['dlogbeta_std'] = np.nan
            self.metadata_df['dlogbeta_lower'] = np.nan
            self.metadata_df['dlogbeta_upper'] = np.nan

            # set confidence intervals for uncertainty
            alpha = 1 - self.confidence_level
            lower_percentile = 100 * (alpha / 2)
            upper_percentile = 100 * (1 - alpha / 2)

        for i in trange(nevents, desc="Computing corrected logbeta"):
            # t0 = time.time()
            eid = eids[i]

            # Store all entries of target event in md_t
            eid_inds = self.metadata_df['_eid'] == eid
            md_t = self.metadata_df[eid_inds]
            md_t = md_t.sort_values(by='_sid') # is this necessary?

            # Store values we are about to use
            edep = md_t['edep'].values[0]
            emag = md_t['emag'].values[0]
            ex = md_t['ex'].values[0]
            ey = md_t['ey'].values[0]

            # Make a copy of the calibration event records DataFrame
            md_c = self.metadata_calib.copy()

            # Filter out calibration event records that:
            #   1) are too shallow or too deep
            #   2) don't share stations with the target event
            filt1 = np.all([
                md_c['edep']>=edep-self.calib_zdist_max, 
                md_c['edep']<=edep+self.calib_zdist_max,
                np.isin(md_c['_sid'].values, md_t['_sid'])
                ], axis=0)
            md_c = md_c[filt1].reset_index(drop=True)

            # Now, md_c contains records of calibration events in the correct 
            # depth range and with the same stations as the target event

            # rough square filter to avoid computing distances for all 
            # calibration events (should be faster than computing distances
            # for all events)
            filt2 = np.all([
                np.abs(md_c['ex'] - ex) <= self.calib_hdist_max*1000.0,
                np.abs(md_c['ey'] - ey) <= self.calib_hdist_max*1000.0,
            ], axis=0)
            md_c = md_c[filt2].reset_index(drop=True)

            # compute station-event distance of remaining calib events
            # simple c = sqrt(a**2 + b**2) -- will need to change for larger regions
            filt3 = np.sqrt((ex - md_c['ex'].values)**2 + (ey - md_c['ey'].values)**2) <= self.calib_hdist_max*1000
            md_c = md_c[filt3].reset_index(drop=True)

            # # simplify md_c by removing unnecessary columns
            # md_c = md_c[['station_name','_eid', '_sid', 'logbeta']]

            ncalib = len(md_c)

            
            # only compute dlogbeta if there are enough remaining calibration events
            if ncalib >= self.n_calib_min:

                # t_pre_test = time.time() - t0

                # Store stations that record target event
                # t_sid_all = np.unique(md_t['_sid'].values)
                t_sid_all = md_t['_sid'].values
                # Store stations that record calibration events (some repeated)
                c_sid_all = md_c['_sid'].values

                # Get indices of calibration events that share a station with target event
                c_ind = np.where(np.isin(c_sid_all, t_sid_all))[0]
                
                # now, all entries in md_c can be used in computing dlogbeta
                # pandas method (slower)
                # md_c = md_c.iloc[c_ind].reset_index(drop=True)
                # c_sid = md_c['_sid'].values
                
                # numpy method
                c_sid = c_sid_all[c_ind]

                # filter out entries in md_t that don't have a station in md_c
                t_ind = np.where(np.isin(t_sid_all, c_sid))[0]
                
                # # pandas
                # md_t = md_t.iloc[t_ind].reset_index(drop=True)
                # t_sid = md_t['_sid'].values
                
                # numpy
                t_sid = t_sid_all[t_ind]
                
                # assert len(t_sid) == len(np.unique(t_sid))

                # Match order
                v = np.searchsorted(t_sid, c_sid)
                
                # # using pandas
                # t_logbeta = md_t['logbeta'].values[v]
                # c_logbeta = md_c['logbeta'].values

                # using numpy
                t_logbeta = md_t['logbeta'].values[t_ind][v]
                c_logbeta = md_c['logbeta'].values[c_ind]

                differences = t_logbeta - c_logbeta
                
                self.metadata_df.loc[eid_inds, 'dlogbeta'] = np.median(differences)

                if self.compute_uncertainty:
                    bootstrap_estimates = np.zeros(self.n_bootstrap)
                    n_samples = len(differences)
                    for b in range(self.n_bootstrap):
                        resample_indices = np.random.choice(
                            n_samples, size=n_samples, replace=True)
                        resampled_differences = differences[resample_indices]
                        bootstrap_estimates[b] = np.median(resampled_differences)
                    
                    self.metadata_df.loc[eid_inds, 'dlogbeta_std'] = np.std(bootstrap_estimates)
                    self.metadata_df.loc[eid_inds, 'dlogbeta_lower'] = np.percentile(
                        bootstrap_estimates, lower_percentile)
                    self.metadata_df.loc[eid_inds, 'dlogbeta_upper'] = np.percentile(
                        bootstrap_estimates, upper_percentile)
                    
                    # plt.figure(figsize=(10,5))
                    # plt.hist(bootstrap_estimates, bins=np.linspace(-1.5, 1.5, 500))
                    # plt.xlabel('dlogbeta')
                    # plt.ylabel('Count')
                    # plt.title(f'dlogbeta for event {eid} std = {np.std(bootstrap_estimates)}')
                    # plt.show()
                    # if i >= 20: raise ValueError()


        # drop rows with NaN dlogbeta
        self.metadata_df = self.metadata_df.dropna(subset=['dlogbeta']).reset_index(drop=True)

    def apply_magnitude_correction(self, corr_type='smoothedspline'):
        self.group_events()
        mag_corr_dM = self.mag_corr_dM
        if corr_type == 'smoothedspline':
            from scipy.signal import savgol_filter

            # Automatically determine the range of magnitude bins
            M_range = np.round((
                np.floor(np.min(self.metadata_ev['emag'].values) / mag_corr_dM) * mag_corr_dM, 
                np.ceil(np.max(self.metadata_ev['emag'].values) / mag_corr_dM) * mag_corr_dM
                ), 4) # fixes rounding errors
            
            # Define the edges and midpoints of the magnitude bins
            edges = np.arange(M_range[0], M_range[1]+1*mag_corr_dM, mag_corr_dM)
            midpoints = (edges[:-1] + edges[1:])/2

            # drop events with emag lower than the first bin (interpolation issue?)
            self.metadata_ev = self.metadata_ev[self.metadata_ev['emag']>=midpoints[0]].reset_index(drop=True)

            # Assign each target event to a magnitude bin
            bin_inds = np.digitize(self.metadata_ev['emag'].values, edges) - 1

            # Compute the median dlogbeta for each magnitude bin
            median_dlbeta = np.empty(len(edges)-1) * np.nan
            # print('computing median_dlbeta')
            for i in range(len(median_dlbeta)):
                median_dlbeta[i] = np.median(self.metadata_ev['dlogbeta'].values[bin_inds==i])

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
            corrections = np.interp(self.metadata_ev['emag'].values, realmids, median_dlbeta_smooth)
            self.correction_mags = realmids
            self.correction_function = median_dlbeta_smooth
            self.metadata_ev['dlogbeta_corr'] = self.metadata_ev['dlogbeta'].values - corrections

            # print("median_dlbeta[~naninds]: ", median_dlbeta[~naninds])
            # print("num nan in emag: ", np.sum(np.isnan(self.metadata_ev['emag'].values)))
            # print("realmids: ", realmids)
            # print("median_dlbeta_smooth: ", median_dlbeta_smooth)
            # print('num nan in corrections: ', np.sum(np.isnan(corrections)))
        
        self.pair_dep.append('dlogbeta_corr')
        self.explode_events()
        self.group_stations()

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
            fields=['event_name', 'emag', 'elon', 'elat', 'edep', 'dlogbeta', 'dlogbeta_std', 'dlogbeta_lower', 'dlogbeta_upper', 'dlogbeta_corr']
        
        np.savetxt(
            filename,
            self.metadata_ev[fields].values,
            fmt=fmt
        )

    def group_events(self):
        print("group_events()")
        # group self.metadata_df by event dependent fields
        self.compute_column_dependencies()
        self.metadata_ev = self.metadata_df.groupby(self.ev_dep, as_index=False)[self.st_dep+self.pair_dep].agg(list)

    def group_stations(self):
        print("group_stations()")
        # group self.metadata_df by station dependent fields
        self.compute_column_dependencies()
        self.metadata_st = self.metadata_df.groupby(self.st_dep, as_index=False, dropna=False)[self.ev_dep+self.pair_dep].agg(list)

    def explode_events(self):
        print('explode_events()')
        self.compute_column_dependencies()
        self.metadata_df = self.metadata_ev.explode(self.st_dep+self.pair_dep)

    def explode_stations(self):
        print('explode_stations()')
        self.compute_column_dependencies()
        self.metadata_df = self.metadata_st.explode(self.ev_dep+self.pair_dep)

    def compute_column_dependencies(self):
        # Columns in self.metadata_df are 'dependent' in one of three ways:
        #     1) event-dependent (e.g. EQ magnitude, EQ location, etc.)
        #     2) station-dependent (e.g. station name, station location, etc.)
        #     3) event-station pair-dependent (e.g. event-station distance, etc.)
        # 
        # This function checks the columns and updates ev_dep, st_dep, pair_dep
        # class variables.
        ev_dep_columns = [
            'event_name', 'emag', 'elat', 'elon', 'edep', '_eid', 'event_id', 
            'evid', 'event', 'ex', 'ey', 'nts', 'dlogbeta', 'dlogbeta_corr',
            'nrec', 'dlogbeta_std', 'dlogbeta_lower', 'dlogbeta_upper'] # fix nts later
        sta_dep_columns = [
            'station_name', 'slat', 'slon', 'selev', '_sid', 'sx', 'sy', 'kappa0'
            ]
        pair_dep_columns = [
            'logbeta', 'deldist', 'a2_max', 's1', 's2', 'stn'
            ]
        columns = self.metadata_df.columns
        self.ev_dep = []
        self.st_dep = []
        self.pair_dep = []

        for col in columns:
            if col in ev_dep_columns:
                self.ev_dep.append(col)
            elif col in sta_dep_columns:
                self.st_dep.append(col)
            elif col in pair_dep_columns:
                self.pair_dep.append(col)
            else:
                raise Warning("Column {} not recognized".format(col))


    def validate_input(self):
        columns = self.metadata_df.columns
        # rename columns
        # check for an event name column
        event_name_columns = ['event_id', 'evid', 'event']
        if 'event_name' not in columns:
            for col in event_name_columns:
                if col in columns:
                    self.metadata_df.rename(columns={col: 'event_name'}, inplace=True)
                    print("Renamed column {} to event_name".format(col))
                    break
        
        # check for a station name column
        station_name_columns = ['station_id', 'sid', 'station', 'stname', 'st_name', 'stid']
        if 'station_name' not in columns:
            for col in station_name_columns:
                if col in columns:
                    self.metadata_df.rename(columns={col: 'station_name'}, inplace=True)
                    print("Renamed column {} to station_name".format(col))
                    break
        
        # change qmag, qlat, qlon, qdep into emag, elat, elon, edep

        if 'qmag' in columns:
            self.metadata_df.rename(columns={'qmag': 'emag'}, inplace=True)

        if 'qlat' in columns:
            self.metadata_df.rename(columns={'qlat': 'elat'}, inplace=True)

        if 'qlon' in columns:
            self.metadata_df.rename(columns={'qlon': 'elon'}, inplace=True)

        if 'qdep' in columns:
            self.metadata_df.rename(columns={'qdep': 'edep'}, inplace=True)


        required_columns = [
            'event_name', 'station_name', 'emag', 'elat', 'elon', 'edep', 
            'slat', 'slon', 'selev', 'deldist']
        for col in required_columns:
            if col not in self.metadata_df.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(self.metadata_df) != self.spectra.shape[0]:
            raise ValueError("Metadata and spectra row counts do not match.")

        self.nrecords_initial = len(self.metadata_df)
        self.nevents_initial = len(self.metadata_df['event_name'].unique())
        self.nstations_initial = len(self.metadata_df['station_name'].unique())

        if self.save_dir is None: self.save_dir = 'tmp/'
        os.makedirs(self.save_dir, exist_ok=True)

        if not self.quiet: self.print_metadata_information()


    def digitize_names(self):
        self.metadata_df, self.event_name_dict = _get_id_from_column(self.metadata_df, 'event_name', '_eid')
        self.metadata_df, self.station_name_dict = _get_id_from_column(self.metadata_df, 'station_name', '_sid')

    def compute_logbeta(self):
        # spectra is an (N x nf) array
        low_inds = self.low_window_inds
        high_inds = self.high_window_inds
        
        low_band = np.median(np.log10(self.spectra[:, low_inds[0]:low_inds[1]+1]), axis=1)
        high_band = np.median(np.log10(self.spectra[:, high_inds[0]:high_inds[1]+1]), axis=1)
        self.metadata_df['logbeta'] = high_band - low_band
        del self.spectra
    
    def compute_eastings_northings(self):
        # this will need to be improved for a larger area
        self.metadata_df['sx'], self.metadata_df['sy'], zn, zl = utm.from_latlon(
            self.metadata_df['slat'].values, 
            self.metadata_df['slon'].values
            )
        self.metadata_df['ex'], self.metadata_df['ey'], zn, zl = utm.from_latlon(
            self.metadata_df['elat'].values, 
            self.metadata_df['elon'].values
            )

    def compute_frequency_bands(self):
        self.low_window_inds = _get_inds_of_values_in_array(self.f, self.low_window_desired)
        self.high_window_inds = _get_inds_of_values_in_array(self.f, self.high_window_desired)

        self.low_window = [self.f[self.low_window_inds[0]], self.f[self.low_window_inds[1]]]
        self.high_window = [self.f[self.high_window_inds[0]], self.f[self.high_window_inds[1]]]
        if not self.quiet: self.print_frequency_information()


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
        nev = len(self.metadata_df['event_name'].unique())
        nrec = len(self.metadata_df)
        print("")
        print("CALIBRATION INFORMATION")
        print("----------------------------")
        print(f"Calibration range:   M {self.calib_mag_range[0]:.2f} to {self.calib_mag_range[1]:.2f}")
        print(f"Calibration events:  {ncalib:,} ({ncalib/nev*100:.2f}%)")
        print(f"Calibration records: {ncalibrec:,} ({ncalibrec/nrec*100:.2f}%)")
        print("")

    def print_metadata_information(self):
        nev = len(self.metadata_df['event_name'].unique())
        nst = len(self.metadata_df['station_name'].unique())
        nrec = len(self.metadata_df)
        print("")
        print("METADATA INFORMATION")
        print("----------------------------")
        if nev==self.nevents_initial:
            print(f"Events:   {nev:,}")
        else:
            print(f"Events:   {nev:,} of {self.nevents_initial:,} inital ({nev/self.nevents_initial*100:.2f}%)")

        if nst==self.nstations_initial:
            print(f"Stations: {nst:,}")
        else:
            print(f"Stations: {nst:,} of {self.nstations_initial:,} inital ({nst/self.nstations_initial*100:.2f}%)")
        
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


