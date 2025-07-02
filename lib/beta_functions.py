import numpy as np
import pandas as pd
import struct



def read_beta_files(beta_dir):
    """Read in all binary .beta files in a directory and return a DataFrame 
    containing all data, with one line per station.

    Args:
        beta_dir (str): Directory containing .beta files

    Returns:
        DataFrame: Pandas DataFrame containing station-grouped information.

    Sources:

    Last Modified:
        2024-07-22
    """
    import os

    beta_files = [el for el in os.listdir(beta_dir) if el.endswith('.beta')]
    beta_files.sort()
    nfiles = len(beta_files)

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

    for i in range(nfiles):
        filepath = beta_dir + beta_files[i]

        f = open(filepath, 'rb')
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        junk = struct.unpack('i', f.read(4))[0]
        FMT_VERSION = struct.unpack('i', f.read(4))[0]

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
    return df

def write_beta_files(df, output_dir, ):
    """Reciprocal function to read_beta_files(). Writes .beta files to a 
    directory.

    Args:
        df (DataFrame): contains station-grouped data
        output_dir (str): Path to save the binary .beta files

    Last Modified:
        2024-07-22
    """

    prec_wf='float32'
    prec_bp='int32'

    # Increment this if the format changes in the future
    FMT_VERSION = int(0)

    for index, row in df.iterrows():
        filepath = output_dir + row['stname'] + ".beta"

        nevents = len(row['evids'])

        with open(filepath, 'wb') as f:
            # Write station header info
            f.write(struct.pack('i', 0))  # Junk
            f.write(struct.pack('i', FMT_VERSION))
            f.write(struct.pack('20s', row['stname'].encode('UTF-8')))
            f.write(struct.pack('f', row['slat']))
            f.write(struct.pack('f', row['slon']))
            f.write(struct.pack('f', row['selev']))
            f.write(struct.pack('30s', row['beta_method'].encode('UTF-8')))
            f.write(struct.pack('30s', row['stn_method'].encode('UTF-8')))
            f.write(struct.pack('f', row['low_f_band'][0]))
            f.write(struct.pack('f', row['low_f_band'][1]))
            f.write(struct.pack('f', row['high_f_band'][0]))
            f.write(struct.pack('f', row['high_f_band'][1]))
            f.write(struct.pack('i', nevents))

            # Write a 4-byte spacer
            f.write(struct.pack('i', 0))


            # Write event data
            for i in range(nevents):

                f.write(struct.pack('ii', 3, 1))  # Junk
                f.write(struct.pack('20s', str(row['evids'][i]).encode('UTF-8')))
                f.write(struct.pack('f', row['qmag'][i]))
                f.write(struct.pack('f', row['qlat'][i]))
                f.write(struct.pack('f', row['qlon'][i]))
                f.write(struct.pack('f', row['qdep'][i]))
                f.write(struct.pack('f', row['beta'][i]))
                f.write(struct.pack('f', row['stn'][i]))
                f.write(struct.pack('f', row['deldist'][i]))

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
    ehead['efslabel'] = f.read(40).decode('UTF-8')
    ehead['datasource'] = f.read(40).decode('UTF-8')
    ehead['maxnumts'] = struct.unpack('i', f.read(4))[0]
    ehead['numts'] = struct.unpack('i', f.read(4))[0]
    ehead['cuspid'] = struct.unpack('i', f.read(4))[0]
    ehead['qtype'] = f.read(4).decode('UTF-8')
    ehead['qmag1type'] = f.read(4).decode('UTF-8')
    ehead['qmag2type'] = f.read(4).decode('UTF-8')
    ehead['qmag3type'] = f.read(4).decode('UTF-8')
    ehead['qmomenttype'] = f.read(4).decode('UTF-8')
    ehead['qlocqual'] = f.read(4).decode('UTF-8')
    ehead['qfocalqual'] = f.read(4).decode('UTF-8')
    ehead['qlat'] = struct.unpack('f', f.read(4))[0]
    ehead['qlon'] = struct.unpack('f', f.read(4))[0]
    ehead['qdep'] = struct.unpack('f', f.read(4))[0]
    ehead['qsc'] = struct.unpack('f', f.read(4))[0]
    ehead['qmag1'] = struct.unpack('f', f.read(4))[0]
    ehead['qmag2'] = struct.unpack('f', f.read(4))[0]
    ehead['qmag3'] = struct.unpack('f', f.read(4))[0]
    ehead['qmoment'] = struct.unpack('f', f.read(4))[0]
    ehead['qstrike'] = struct.unpack('f', f.read(4))[0]
    ehead['qdip'] = struct.unpack('f', f.read(4))[0]
    ehead['qrake'] = struct.unpack('f', f.read(4))[0]
    ehead['qyr'] = struct.unpack('i', f.read(4))[0]
    ehead['qmon'] = struct.unpack('i', f.read(4))[0]
    ehead['qdy'] = struct.unpack('i', f.read(4))[0]
    ehead['qhr'] = struct.unpack('i', f.read(4))[0]
    ehead['qmn'] = struct.unpack('i', f.read(4))[0]

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
