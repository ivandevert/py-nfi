import numpy as np
import pandas as pd
import struct



def read_beta_files(beta_dir):
    """Read in all binary .beta files in a directory and return a DataFrame 
    containing all data, with one line per station.

    not true 

    Args:
        beta_dir (str): Directory containing .beta files

    Returns:
        DataFrame: Pandas DataFrame containing station-grouped information.

    Sources:

    Last Modified:
        2024-10-08
    """
    import os
    try:
        from tqdm import tqdm, trange
        loopfunc = trange
    except:
        loopfunc = range

    beta_files = [el for el in os.listdir(beta_dir) if el.endswith('.beta')]
    beta_files.sort()
    nfiles = len(beta_files)
    print(f"{nfiles} beta files found")

    # get format of files
    filepath = beta_dir + beta_files[0]
    f = open(filepath, 'rb')
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0, 0)
    _ = struct.unpack('i', f.read(4))[0]
    FMT_VERSION = struct.unpack('i', f.read(4))[0]
    f.close()

    if FMT_VERSION==0:
        df = pd.DataFrame({
            "stname": pd.arrays.SparseArray(dtype="str", data=[]), 
            "slat": pd.arrays.SparseArray(dtype="float", data=[]), 
            "slon": pd.arrays.SparseArray(dtype="float", data=[]), 
            "selev": pd.arrays.SparseArray(dtype="float", data=[]), 

            "event_id": pd.arrays.SparseArray(dtype="object", data=[]), 
            "qmag": pd.arrays.SparseArray(dtype="object", data=[]), 
            "qlat": pd.arrays.SparseArray(dtype="object", data=[]), 
            "qlon": pd.arrays.SparseArray(dtype="object", data=[]), 
            "qdep": pd.arrays.SparseArray(dtype="object", data=[]), 
            "beta": pd.arrays.SparseArray(dtype="object", data=[]), 
            "stn": pd.arrays.SparseArray(dtype="object", data=[]), 
            "deldist": pd.arrays.SparseArray(dtype="object", data=[]), 
        })

    if FMT_VERSION==1:
        df = pd.DataFrame({
            "event_id": pd.arrays.SparseArray(dtype="str", data=[]), 
            "qmag": pd.arrays.SparseArray(dtype="float", data=[]), 
            "qlat": pd.arrays.SparseArray(dtype="float", data=[]), 
            "qlon": pd.arrays.SparseArray(dtype="float", data=[]), 
            "qdep": pd.arrays.SparseArray(dtype="float", data=[]), 
            "nstations": pd.arrays.SparseArray(dtype="int", data=[]), 

            "stname": pd.arrays.SparseArray(dtype="object", data=[]), 
            "slat": pd.arrays.SparseArray(dtype="object", data=[]), 
            "slon": pd.arrays.SparseArray(dtype="object", data=[]), 
            "selev": pd.arrays.SparseArray(dtype="object", data=[]), 
            "logbeta": pd.arrays.SparseArray(dtype="object", data=[]), 
            "stn": pd.arrays.SparseArray(dtype="object", data=[]), 
            "deldist": pd.arrays.SparseArray(dtype="object", data=[]), 
        })
    
    for i in loopfunc(nfiles):
        filepath = beta_dir + beta_files[i]

        f = open(filepath, 'rb')
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        junk = struct.unpack('i', f.read(4))[0]
        FMT_VERSION = struct.unpack('i', f.read(4))[0]

        # station-organized
        if FMT_VERSION == 0:
            stname = f.read(20).decode('UTF-8').strip('\x00')
            slat = struct.unpack('f', f.read(4))[0]
            slon = struct.unpack('f', f.read(4))[0]
            selev = struct.unpack('f', f.read(4))[0]
            beta_method = f.read(30).decode('UTF-8')
            stn_method = f.read(30).decode('UTF-8')
            low_f_band = (struct.unpack('f', f.read(4))[0], 
                          struct.unpack('f', f.read(4))[0])
            high_f_band = (struct.unpack('f', f.read(4))[0],
                           struct.unpack('f', f.read(4))[0])
            nevents = struct.unpack('i', f.read(4))[0]

            evids = np.zeros(nevents, dtype='<U20')
            qmag = np.zeros(nevents, dtype=float)
            qlat = np.zeros_like(qmag)
            qlon = np.zeros_like(qmag)
            qdep = np.zeros_like(qmag)
            beta = np.zeros_like(qmag)
            stn = np.zeros_like(qmag)
            deldist = np.zeros_like(qmag)
            for j in range(nevents):
                junk1 = struct.unpack('i', f.read(4))[0]
                junk2 = struct.unpack('i', f.read(4))[0]
                # print(junk1, junk2)
                evids[j] = f.read(20).decode('UTF-8').strip('\x01\x00')
                junk = struct.unpack('i', f.read(4))[0]
                qmag[j] = struct.unpack('f', f.read(4))[0]
                qlat[j] = struct.unpack('f', f.read(4))[0]
                qlon[j] = struct.unpack('f', f.read(4))[0]
                qdep[j] = struct.unpack('f', f.read(4))[0]
                beta[j] = struct.unpack('f', f.read(4))[0]
                stn[j] = struct.unpack('f', f.read(4))[0]
                deldist[j] = struct.unpack('f', f.read(4))[0]
            df.loc[len(df)] = [stname, slat, slon, selev, evids, qmag, qlat, qlon, qdep, beta, stn, deldist]
        
        # event-organized
        elif FMT_VERSION == 1:
            event_id = f.read(20).decode('UTF-8').strip('\x00')
            qmag = struct.unpack('f', f.read(4))[0]
            qlat = struct.unpack('f', f.read(4))[0]
            qlon = struct.unpack('f', f.read(4))[0]
            qdep = struct.unpack('f', f.read(4))[0]

            beta_method = f.read(30).decode('UTF-8')
            stn_method = f.read(30).decode('UTF-8')
            low_f_band = (struct.unpack('f', f.read(4))[0], 
                          struct.unpack('f', f.read(4))[0])
            high_f_band = (struct.unpack('f', f.read(4))[0],
                           struct.unpack('f', f.read(4))[0])
            nstations = struct.unpack('i', f.read(4))[0]
            # print(f"{nstations} stations")

            stname = np.zeros(nstations, dtype='<U20')
            slat = np.zeros(nstations, dtype=float)
            slon = np.zeros_like(slat)
            selev = np.zeros_like(slat)
            beta = np.zeros_like(slat)
            stn = np.zeros_like(slat)
            deldist = np.zeros_like(slat)
            for j in range(nstations):
                junk1 = struct.unpack('i', f.read(4))[0]
                junk2 = struct.unpack('i', f.read(4))[0]
                # print(junk1, junk2)
                stname[j] = f.read(20).decode('UTF-8').strip('\x01\x00')
                junk = struct.unpack('i', f.read(4))[0]
                slat[j] = struct.unpack('f', f.read(4))[0]
                slon[j] = struct.unpack('f', f.read(4))[0]
                selev[j] = struct.unpack('f', f.read(4))[0]
                beta[j] = struct.unpack('f', f.read(4))[0]
                stn[j] = struct.unpack('f', f.read(4))[0]
                deldist[j] = struct.unpack('f', f.read(4))[0]
            df.loc[len(df)] = [event_id, qmag, qlat, qlon, qdep, nstations, stname, slat, slon, selev, beta, stn, deldist]
        else: raise ValueError("Uh oh")
        f.close()
    return df

def write_beta_files(df, output_dir, ):
    """Reciprocal function to read_beta_files(). Writes .beta files to a 
    directory.

    Now uses event-grouped DataFrame

    Args:
        df (DataFrame): contains event-grouped data
        output_dir (str): Path to save the binary .beta files

    Last Modified:
        2024-10-08
    """

    prec_wf='float32'
    prec_bp='int32'

    # Increment this if the format changes in the future
    FMT_VERSION = int(1)

    for index, row in df.iterrows():
        filepath = output_dir + str(row['event_id']) + ".beta"

        # nevents = len(row['evids'])
        nstations = len(row['stname'])

        with open(filepath, 'wb') as f:
        
            # Write event data
            f.write(struct.pack('i', 0))  # Junk
            f.write(struct.pack('i', FMT_VERSION))
            f.write(struct.pack('20s', str(row['event_id']).encode('UTF-8')))
            f.write(struct.pack('f', row['qmag']))
            f.write(struct.pack('f', row['qlat']))
            f.write(struct.pack('f', row['qlon']))
            f.write(struct.pack('f', row['qdep']))

            # write parameters
            f.write(struct.pack('30s', row['beta_method'].encode('UTF-8')))
            f.write(struct.pack('30s', row['stn_method'].encode('UTF-8')))
            f.write(struct.pack('f', row['low_f_band'][0]))
            f.write(struct.pack('f', row['low_f_band'][1]))
            f.write(struct.pack('f', row['high_f_band'][0]))
            f.write(struct.pack('f', row['high_f_band'][1]))
            f.write(struct.pack('i', nstations))

            # Write a 4-byte spacer
            f.write(struct.pack('i', 0))

            for j in range(nstations):

                # Write station header info
                f.write(struct.pack('ii', 3, 1))  # Junk
                f.write(struct.pack('20s', row['stname'][j].encode('UTF-8')))
                f.write(struct.pack('f', row['slat'][j]))
                f.write(struct.pack('f', row['slon'][j]))
                f.write(struct.pack('f', row['selev'][j]))
                f.write(struct.pack('f', row['beta'][j]))
                f.write(struct.pack('f', row['stn'][j]))
                f.write(struct.pack('f', row['deldist'][j]))
                # f.write(struct.pack('i', nevents))

                # Write a 4-byte spacer
                f.write(struct.pack('i', 0))





def read_spec(filepath, prec_wf='float32', prec_bp='int32'):
    """Function to read .spec files outputted by Peter Shearer's fortran
    codes. Copied from stressdrop_file_IO.py on 2024-06-25

    Args:
        filepath (str): Path of the .spec file to read

    Returns:
        shead: Spectral method header info
        ehead: Event header info
        spec: Spectra

    Sources:

    Last Modified:
        2024-06-25
    """

    f = open(filepath, 'rb')
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0, 0)

    spec = []

    shead = {}
    ehead = {}

    junk = struct.unpack('i', f.read(4))[0]
    shead['ispec_method'] = struct.unpack('i', f.read(4))[0]
    shead['ntwind'] = struct.unpack('i', f.read(4))[0]
    shead['nf'] = struct.unpack('i', f.read(4))[0]
    shead['twindoff'] = struct.unpack('f', f.read(4))[0]
    shead['dt'] = struct.unpack('f', f.read(4))[0]
    shead['df'] = struct.unpack('f', f.read(4))[0]    

    junk1 = struct.unpack('i', f.read(4))[0]
    junk2 = struct.unpack('i', f.read(4))[0]
    ehead['efslabel']       = f.read(40).decode('UTF-8')
    ehead['datasource']     = f.read(40).decode('UTF-8')
    ehead['maxnumts']       = struct.unpack('i', f.read(4))[0]
    ehead['numts']          = struct.unpack('i', f.read(4))[0]
    ehead['cuspid']         = struct.unpack('i', f.read(4))[0]
    ehead['qtype']          = f.read(4).decode('UTF-8')
    ehead['qmag1type']      = f.read(4).decode('UTF-8')
    ehead['qmag2type']      = f.read(4).decode('UTF-8')
    ehead['qmag3type']      = f.read(4).decode('UTF-8')
    ehead['qmomenttype']    = f.read(4).decode('UTF-8')
    ehead['qlocqual']       = f.read(4).decode('UTF-8')
    ehead['qfocalqual']     = f.read(4).decode('UTF-8')
    ehead['qlat']           = struct.unpack('f', f.read(4))[0]
    ehead['qlon']           = struct.unpack('f', f.read(4))[0]
    ehead['qdep']           = struct.unpack('f', f.read(4))[0]
    ehead['qsc']            = struct.unpack('f', f.read(4))[0]
    ehead['qmag1']          = struct.unpack('f', f.read(4))[0]
    ehead['qmag2']          = struct.unpack('f', f.read(4))[0]
    ehead['qmag3']          = struct.unpack('f', f.read(4))[0]
    ehead['qmoment']        = struct.unpack('f', f.read(4))[0]
    ehead['qstrike']        = struct.unpack('f', f.read(4))[0]
    ehead['qdip']           = struct.unpack('f', f.read(4))[0]
    ehead['qrake']          = struct.unpack('f', f.read(4))[0]
    ehead['qyr']            = struct.unpack('i', f.read(4))[0]
    ehead['qmon']           = struct.unpack('i', f.read(4))[0]
    ehead['qdy']            = struct.unpack('i', f.read(4))[0]
    ehead['qhr']            = struct.unpack('i', f.read(4))[0]
    ehead['qmn']            = struct.unpack('i', f.read(4))[0]

    # 20 4-byte fields reserved for future uses - skip (80 bytes)
    for idum in range(0, 20):
        dummy = struct.unpack('i', f.read(4))[0]

    # Now loop over all the time series
    for ii in range(0, ehead['numts']):
        # Assemble tshead
        tshead = {}
        junk1 = struct.unpack('i', f.read(4))[0]
        junk2 = struct.unpack('i', f.read(4))[0]
        
        tshead['stname'] = f.read(8).decode('UTF-8')
        tshead['loccode'] = f.read(8).decode('UTF-8')
        tshead['datasource'] = f.read(8).decode('UTF-8')
        tshead['sensor'] = f.read(8).decode('UTF-8')
        tshead['units'] = f.read(8).decode('UTF-8')
        tshead['chnm'] = f.read(4).decode('UTF-8')
        tshead['stype'] = f.read(4).decode('UTF-8')
        tshead['dva'] = f.read(4).decode('UTF-8')
        tshead['pick1q'] = f.read(4).decode('UTF-8')
        tshead['pick2q'] = f.read(4).decode('UTF-8')
        tshead['pick3q'] = f.read(4).decode('UTF-8')
        tshead['pick4q'] = f.read(4).decode('UTF-8')
        tshead['pick1name'] = f.read(4).decode('UTF-8')
        tshead['pick2name'] = f.read(4).decode('UTF-8')
        tshead['pick3name'] = f.read(4).decode('UTF-8')
        tshead['pick4name'] = f.read(4).decode('UTF-8')
        tshead['ppolarity'] = f.read(4).decode('UTF-8')
        tshead['problem'] = f.read(4).decode('UTF-8')
        tshead['npts'] = struct.unpack('i', f.read(4))[0]
        tshead['syr'] = struct.unpack('i', f.read(4))[0]
        tshead['smon'] = struct.unpack('i', f.read(4))[0]
        tshead['sdy'] = struct.unpack('i', f.read(4))[0]
        tshead['shr'] = struct.unpack('i', f.read(4))[0]
        tshead['smn'] = struct.unpack('i', f.read(4))[0]
        tshead['compazi'] = struct.unpack('f', f.read(4))[0]
        tshead['compang'] = struct.unpack('f', f.read(4))[0]
        tshead['gain'] = struct.unpack('f', f.read(4))[0]
        tshead['f1'] = struct.unpack('f', f.read(4))[0]
        tshead['f2'] = struct.unpack('f', f.read(4))[0]
        tshead['dt'] = struct.unpack('f', f.read(4))[0]
        tshead['ssc'] = struct.unpack('f', f.read(4))[0]
        tshead['tdif'] = struct.unpack('f', f.read(4))[0]
        tshead['slat'] = struct.unpack('f', f.read(4))[0]
        tshead['slon'] = struct.unpack('f', f.read(4))[0]
        tshead['selev'] = struct.unpack('f', f.read(4))[0]
        tshead['deldist'] = struct.unpack('f', f.read(4))[0]
        tshead['sazi'] = struct.unpack('f', f.read(4))[0]
        tshead['qazi'] = struct.unpack('f', f.read(4))[0]
        tshead['pick1'] = struct.unpack('f', f.read(4))[0]
        tshead['pick2'] = struct.unpack('f', f.read(4))[0]
        tshead['pick3'] = struct.unpack('f', f.read(4))[0]
        tshead['pick4'] = struct.unpack('f', f.read(4))[0]

        # 20 4-byte fields reserved for future uses - skip
        for idum in range(0, 22):
            dummy = struct.unpack('i', f.read(4))[0]

        # Read the windowed data and spectra
        x1out = np.fromfile(f, dtype = prec_wf, count = shead['ntwind'])
        x2out = np.fromfile(f, dtype = prec_wf, count = shead['ntwind'])

        s1out = np.fromfile(f, dtype = prec_wf, count = shead['nf'])
        s2out = np.fromfile(f, dtype = prec_wf, count = shead['nf'])

        # Bundle tsheader and time-series for this waveform into efsdata, then append to list
        tshead['x1'] = x1out
        tshead['x2'] = x2out
        tshead['s1'] = s1out
        tshead['s2'] = s2out
        spec.append(tshead)

        # some have fewer spectra than numts for some reason
        if file_size - f.tell() < 10: break

    f.close()

    ehead['numts'] = len(spec)

    return shead, ehead, spec

def read_spec_df(filepath):
    """ only read certain fields. Need:
    
    event id, numts (actual, not ehead['numts']), qlat, qlon, qdep, qmag, qmag1type, spectra

    """

    f = open(filepath, 'rb')
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0, 0)

    spec = []

    # read in file header info
    f.seek(8, 0)
    ntwind, nf = struct.unpack('2i', f.read(8))
    f.seek(4,1)
    dt, df = struct.unpack('2f', f.read(8))

    # bytepos=28
    
    # bytes in one object in spec
    tsbytes = (284 + 8 * (ntwind+nf))
    nts = int(np.floor((file_size-300) / tsbytes))


    f.seek(96, 1)
    event_id = struct.unpack('i', f.read(4))[0]
    f.seek(4, 1)
    qmagtype = struct.unpack('4s', f.read(4))[0].decode('UTF-8')
    f.seek(20, 1)
    qlat, qlon, qdep = struct.unpack('3f', f.read(12))
    f.seek(4, 1)
    qmag = struct.unpack('f', f.read(4))[0]

    # bytepos=176

    f.seek(44+80, 1)
    # should be at bytepos=300 here

    # print(ntwind, nf, dt, df, nts, event_id, qmagtype, qlat, qlon, qdep, qmag)

    X1 = [[]] * nts
    X2 = [[]] * nts
    S1 = [[]] * nts
    S2 = [[]] * nts
    stid = [[]] * nts
    slat = [[]] * nts
    slon = [[]] * nts
    selev = [[]] * nts
    deldist = [[]] * nts

    ntfmt = f'{ntwind}f'
    nffmt = f'{nf}f'
    for i in range(nts):
        f.seek(8, 1)

        stname = f.read(8).decode('UTF-8').strip()
        loccode = f.read(8).decode('UTF-8').strip()
        f.seek(24, 1)
        chnm = f.read(4).decode('UTF-8').strip()
        net = f.read(4).decode('UTF-8').strip()

        stid[i] = '.'.join([net, stname, loccode, chnm])

        # print(stid)

        f.seek(100,1)
        slat[i], slon[i], selev[i], deldist[i] = struct.unpack('4f', f.read(16))
        f.seek(112+ntwind*8, 1)
        # X1[i] = np.array(struct.unpack(ntfmt, f.read(ntwind*4)), dtype=float)
        # X2[i] = np.array(struct.unpack(ntfmt, f.read(ntwind*4)), dtype=float)
        S1[i] = np.array(struct.unpack(nffmt, f.read(nf*4)), dtype=float)
        S2[i] = np.array(struct.unpack(nffmt, f.read(nf*4)), dtype=float)
    
    D = {
        'event_id':     event_id,
        'nts':          nts,
        'qlat':         qlat,
        'qlon':         qlon,
        'qdep':         qdep,
        'qmag':         qmag,
        'stid':         stid,
        'slat':         slat,
        'slon':         slon,
        'selev':        selev,
        'deldist':      deldist,
        's1':           S1,
        's2':           S2
    }

    df = pd.DataFrame(D)

    return df

def read_spec_max_df(filepath):
    """ only read certain fields. Need:
    
    event id, numts (actual, not ehead['numts']), qlat, qlon, qdep, qmag, qmag1type, spectra

    this also computes max amplitude - min amplitude of seismograms
    """

    f = open(filepath, 'rb')
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0, 0)

    spec = []

    # read in file header info
    f.seek(8, 0)
    ntwind, nf = struct.unpack('2i', f.read(8))
    f.seek(4,1)
    dt, df = struct.unpack('2f', f.read(8))

    # bytepos=28
    
    # bytes in one object in spec
    tsbytes = (284 + 8 * (ntwind+nf))
    nts = int(np.floor((file_size-300) / tsbytes))


    f.seek(96, 1)
    event_id = struct.unpack('i', f.read(4))[0]
    f.seek(4, 1)
    qmagtype = struct.unpack('4s', f.read(4))[0].decode('UTF-8')
    f.seek(20, 1)
    qlat, qlon, qdep = struct.unpack('3f', f.read(12))
    f.seek(4, 1)
    qmag = struct.unpack('f', f.read(4))[0]

    # bytepos=176

    f.seek(44+80, 1)
    # should be at bytepos=300 here

    # print(ntwind, nf, dt, df, nts, event_id, qmagtype, qlat, qlon, qdep, qmag)

    # X1 = [[]] * nts
    # X2 = [[]] * nts
    X2_max = [[]] * nts
    S1 = [[]] * nts
    S2 = [[]] * nts
    stid = [[]] * nts
    slat = [[]] * nts
    slon = [[]] * nts
    selev = [[]] * nts
    deldist = [[]] * nts

    ntfmt = f'{ntwind}f'
    nffmt = f'{nf}f'
    for i in range(nts):
        f.seek(8, 1)

        stname = f.read(8).decode('UTF-8').strip()
        loccode = f.read(8).decode('UTF-8').strip()
        f.seek(24, 1)
        chnm = f.read(4).decode('UTF-8').strip()
        net = f.read(4).decode('UTF-8').strip()

        stid[i] = '.'.join([net, stname, loccode, chnm])

        # print(stid)

        f.seek(100,1)
        slat[i], slon[i], selev[i], deldist[i] = struct.unpack('4f', f.read(16))
        # f.seek(112+ntwind*4, 1)
        f.seek(112+ntwind*4, 1)
        # X1[i] = np.array(struct.unpack(ntfmt, f.read(ntwind*4)), dtype=float)
        X2 = np.diff(np.array(struct.unpack(ntfmt, f.read(ntwind*4)), dtype=float))
        X2_max[i] = np.max(X2) - np.min(X2)
        S1[i] = np.array(struct.unpack(nffmt, f.read(nf*4)), dtype=float)
        S2[i] = np.array(struct.unpack(nffmt, f.read(nf*4)), dtype=float)
        
    D = {
        'event_id':     event_id,
        'nts':          nts,
        'qlat':         qlat,
        'qlon':         qlon,
        'qdep':         qdep,
        'qmag':         qmag,
        'stid':         stid,
        'slat':         slat,
        'slon':         slon,
        'selev':        selev,
        'deldist':      deldist,
        'a2_max':       X2_max,
        's1':           S1,
        's2':           S2
    }

    df = pd.DataFrame(D)

    return df

def write_spec(filepath, ehead, shead, spec, prec_wf='float32', prec_bp='int32'):
    """Reciprocal function to readspec. Writes spectrum and associated 
    information to a .spec file.

    Args:
        filepath (str): Path of the .spec file to generate
        shead (_type_): Spectral method header info dict object. Should
            contain ispec_method, ntwind, nf, twindoff, dt, and df.
        spec (_type_): _description_
        prec_wf (str, optional): _description_. Defaults to 'float32'.
        prec_bp (str, optional): _description_. Defaults to 'int32'.

    Sources:

    Last Modified:
        2023-10-30
    """

    with open(filepath, 'wb') as f:
        # Write junk and shead
        f.write(struct.pack('i', 0))  # Junk
        f.write(struct.pack('i', shead['ispec_method']))
        f.write(struct.pack('i', shead['ntwind']))
        f.write(struct.pack('i', shead['nf']))
        f.write(struct.pack('f', shead['twindoff']))
        f.write(struct.pack('f', shead['dt']))
        f.write(struct.pack('f', shead['df']))
        
        # Write junk and ehead
        f.write(struct.pack('ii', 0, 0))  # Junk
        f.write(struct.pack('40s',  ehead['efslabel'].encode('UTF-8').ljust(40)))
        f.write(struct.pack('40s',  ehead['datasource'].encode('UTF-8').ljust(40)))
        f.write(struct.pack('i',    ehead['maxnumts']))
        f.write(struct.pack('i',    ehead['numts']))
        f.write(struct.pack('i',    ehead['cuspid']))
        f.write(struct.pack('4s',   ehead['qtype'].encode('UTF-8').ljust(4)))
        f.write(struct.pack('4s',   ehead['qmag1type'].encode('UTF-8').ljust(4)))
        f.write(struct.pack('4s',   ehead['qmag2type'].encode('UTF-8').ljust(4)))
        f.write(struct.pack('4s',   ehead['qmag3type'].encode('UTF-8').ljust(4)))
        f.write(struct.pack('4s',   ehead['qmomenttype'].encode('UTF-8').ljust(4)))
        f.write(struct.pack('4s',   ehead['qlocqual'].encode('UTF-8').ljust(4)))
        f.write(struct.pack('4s',   ehead['qfocalqual'].encode('UTF-8').ljust(4)))
        f.write(struct.pack('f',    ehead['qlat']))
        f.write(struct.pack('f',    ehead['qlon']))
        f.write(struct.pack('f',    ehead['qdep']))
        f.write(struct.pack('f',    ehead['qsc']))
        f.write(struct.pack('f',    ehead['qmag1']))
        f.write(struct.pack('f',    ehead['qmag2']))
        f.write(struct.pack('f',    ehead['qmag3']))
        f.write(struct.pack('f',    ehead['qmoment']))
        f.write(struct.pack('f',    ehead['qstrike']))
        f.write(struct.pack('f',    ehead['qdip']))
        f.write(struct.pack('f',    ehead['qrake']))
        f.write(struct.pack('i',    ehead['qyr']))
        f.write(struct.pack('i',    ehead['qmon']))
        f.write(struct.pack('i',    ehead['qdy']))
        f.write(struct.pack('i',    ehead['qhr']))
        f.write(struct.pack('i',    ehead['qmn']))

        # Write 20 4-byte fields reserved for future uses
        for _ in range(20):
            f.write(struct.pack('i', 0))

        # Write time series data
        for wv in spec:
            f.write(struct.pack('ii', 3, 1))  # Junk
            f.write(struct.pack('8s', wv['stname'].encode('UTF-8').ljust(8)))
            f.write(struct.pack('8s', wv['loccode'].encode('UTF-8').ljust(8)))
            f.write(struct.pack('8s', wv['datasource'].encode('UTF-8').ljust(8)))
            f.write(struct.pack('8s', wv['sensor'].encode('UTF-8').ljust(8)))
            f.write(struct.pack('8s', wv['units'].encode('UTF-8').ljust(8)))
            f.write(struct.pack('4s', wv['chnm'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['stype'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['dva'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick1q'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick2q'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick3q'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick4q'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick1name'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick2name'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick3name'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['pick4name'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['ppolarity'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('4s', wv['problem'].encode('UTF-8').ljust(4)))
            f.write(struct.pack('i',  wv['npts']))
            f.write(struct.pack('i',  wv['syr']))
            f.write(struct.pack('i',  wv['smon']))
            f.write(struct.pack('i',  wv['sdy']))
            f.write(struct.pack('i',  wv['shr']))
            f.write(struct.pack('i',  wv['smn']))
            f.write(struct.pack('f',  wv['compazi']))
            f.write(struct.pack('f',  wv['compang']))
            f.write(struct.pack('f',  wv['gain']))
            f.write(struct.pack('f',  wv['f1']))
            f.write(struct.pack('f',  wv['f2']))
            f.write(struct.pack('f',  wv['dt']))
            f.write(struct.pack('f',  wv['ssc']))
            f.write(struct.pack('f',  wv['tdif']))
            f.write(struct.pack('f',  wv['slat']))
            f.write(struct.pack('f',  wv['slon']))
            f.write(struct.pack('f',  wv['selev']))
            f.write(struct.pack('f',  wv['deldist']))
            f.write(struct.pack('f',  wv['sazi']))
            f.write(struct.pack('f',  wv['qazi']))
            f.write(struct.pack('f',  wv['pick1']))
            f.write(struct.pack('f',  wv['pick2']))
            f.write(struct.pack('f',  wv['pick3']))
            f.write(struct.pack('f',  wv['pick4']))

            # Write 22 4-byte fields reserved for future uses
            for _ in range(22):
                f.write(struct.pack('i', 0))

            # Write the waveform data
            np.array(wv['x1'], dtype=prec_wf).tofile(f)
            np.array(wv['x2'], dtype=prec_wf).tofile(f)
            np.array(wv['s1'], dtype=prec_wf).tofile(f)
            np.array(wv['s2'], dtype=prec_wf).tofile(f)

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

### The following files are outdated.
def read_betatxt_files(betatxt_dir, ):
    """Read in all ASCII .betatxt files in a directory and return a DataFrame 
    containing all data, with one line per station. 

    Args:
        betatxt_dir (str): Directory containing .beta files

    Returns:
        DataFrame: Pandas DataFrame containing station-grouped information.

    Sources:

    Last Modified:
        2024-07-22
    """

    df = pd.DataFrame({
        "stname": pd.arrays.SparseArray(dtype="str", data=[]), 
        "slat": pd.arrays.SparseArray(dtype="float", data=[]), 
        "slon": pd.arrays.SparseArray(dtype="float", data=[]), 
        "selev": pd.arrays.SparseArray(dtype="float", data=[]), 

        "event_id": pd.arrays.SparseArray(dtype="object", data=[]), 
        "qmag": pd.arrays.SparseArray(dtype="object", data=[]), 
        "qlat": pd.arrays.SparseArray(dtype="object", data=[]), 
        "qlon": pd.arrays.SparseArray(dtype="object", data=[]), 
        "qdep": pd.arrays.SparseArray(dtype="object", data=[]), 
        "beta": pd.arrays.SparseArray(dtype="object", data=[]), 
        "stn": pd.arrays.SparseArray(dtype="object", data=[]), 
        "deldist": pd.arrays.SparseArray(dtype="object", data=[]), 
    })

    beta_files = [el for el in os.listdir(betatxt_dir) if el.endswith('.betatxt')]
    beta_files.sort()

    nfiles = len(beta_files)

    for j in range(nfiles):
        filepath = betatxt_dir + beta_files[j]

        low_f_band = np.zeros(2, dtype=float)
        high_f_band = np.zeros(2, dtype=float)

        with open(filepath, 'r') as fp:
            line = fp.readline().strip()

        stname, slat, slon, selev, beta_method, stn_method, low_f_band[0], \
        low_f_band[1], high_f_band[0], high_f_band[1], nevents \
        = line.split("  ")

        # print(line)

        column_names = ['evids', 'qmag', 'qlat', 'qlon', 'qdep', 'beta', 'stn', 'deldist']
        data = pd.read_csv(filepath, sep='\\s+', skiprows=1, 
            names=column_names)
        evids = data['evids'].values
        qmag = data['qmag'].values
        qlat = data['qlat'].values
        qlon = data['qlon'].values
        qdep = data['qdep'].values
        beta = data['beta'].values
        stn = data['stn'].values
        deldist = data['deldist'].values
        df.loc[len(df)] = [stname, slat, slon, selev, evids, qmag, qlat, qlon, qdep, beta, stn, deldist]

        # print(data)
    return df

def write_betatxt_files(df, filedir, low_f_band, high_f_band, beta_method, stn_method):
    """Write information to a .betatxt file. Structure:
    1) Station header. Station name, slat, slon, selev, beta_method, stn_method, low_f_0, low_f_1, high_f_0, high_f_1, nevents
    2) Event lines. Event ID, qmag, beta, low_f_value, high_f_value, stn, deldist

    Sources:

    Last Modified:
        2024-07-18
    """

    for j in range(len(df)):

        stname = df.at[j, 'stname']
        slat = df.at[j, 'slat']
        slon = df.at[j, 'slon']
        selev = df.at[j, 'selev']

        evids = df.at[j, 'evids']
        qmag = df.at[j, 'qmag']
        qlat = df.at[j, 'qlat']
        qlon = df.at[j, 'qlon']
        qdep = df.at[j, 'qlon']
        beta = df.at[j, 'beta']
        stn = df.at[j, 'stn']
        deldist = df.at[j, 'deldist']

        nevents = len(evids)

        filepath = filedir + stname + ".betatxt"
        shead_objs = [
            stname, 
            slat, 
            slon, 
            selev, 
            beta_method, 
            stn_method, 
            low_f_band[0],
            low_f_band[1],
            high_f_band[0],
            high_f_band[1],
            nevents
            ]
        # print(" ".join([str(el) for el in shead_objs]))
        with open(filepath, 'w') as fp:
            fp.write("  ".join([str(el) for el in shead_objs])+"\n")

            for i in range(nevents):
                ln = f"{evids[i]} {qmag[i]:.2f} {qlat[i]:13.8f} {qlon[i]:13.8f} {qdep[i]:11.6f} {beta[i]:9.6e} {stn[i]:7.5e} {deldist[i]:8.5e}\n"
                # print(ln)
                fp.write(ln)
    return


# def write_beta_files(df, output_dir, ):
#     """Reciprocal function to read_beta_files(). Writes .beta files to a 
#     directory.

#     Args:
#         df (DataFrame): contains station-grouped data
#         output_dir (str): Path to save the binary .beta files

#     Last Modified:
#         2024-07-22
#     """

#     prec_wf='float32'
#     prec_bp='int32'

#     # Increment this if the format changes in the future
#     FMT_VERSION = int(0)

#     for index, row in df.iterrows():
#         filepath = output_dir + row['stname'] + ".beta"

#         nevents = len(row['evids'])

#         with open(filepath, 'wb') as f:
#             # Write station header info
#             f.write(struct.pack('i', 0))  # Junk
#             f.write(struct.pack('i', FMT_VERSION))
#             f.write(struct.pack('20s', row['stname'].encode('UTF-8')))
#             f.write(struct.pack('f', row['slat']))
#             f.write(struct.pack('f', row['slon']))
#             f.write(struct.pack('f', row['selev']))
#             f.write(struct.pack('30s', row['beta_method'].encode('UTF-8')))
#             f.write(struct.pack('30s', row['stn_method'].encode('UTF-8')))
#             f.write(struct.pack('f', row['low_f_band'][0]))
#             f.write(struct.pack('f', row['low_f_band'][1]))
#             f.write(struct.pack('f', row['high_f_band'][0]))
#             f.write(struct.pack('f', row['high_f_band'][1]))
#             f.write(struct.pack('i', nevents))

#             # Write a 4-byte spacer
#             f.write(struct.pack('i', 0))


#             # Write event data
#             for i in range(nevents):

#                 f.write(struct.pack('ii', 3, 1))  # Junk
#                 f.write(struct.pack('20s', str(row['evids'][i]).encode('UTF-8')))
#                 f.write(struct.pack('f', row['qmag'][i]))
#                 f.write(struct.pack('f', row['qlat'][i]))
#                 f.write(struct.pack('f', row['qlon'][i]))
#                 f.write(struct.pack('f', row['qdep'][i]))
#                 f.write(struct.pack('f', row['beta'][i]))
#                 f.write(struct.pack('f', row['stn'][i]))
#                 f.write(struct.pack('f', row['deldist'][i]))

#                 # Write a 4-byte spacer
#                 f.write(struct.pack('i', 0))
