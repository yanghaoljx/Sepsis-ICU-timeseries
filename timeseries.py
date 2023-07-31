import pandas as pd
import itertools
import numpy as np
import os


def reconfigure_timeseries(timeseries, offset_column, feature_column=None, test=False):
    if test:
        timeseries = timeseries.iloc[0:5000]  # for testing purposes
    timeseries.set_index(['patientunitstayid', pd.to_timedelta(timeseries[offset_column], unit='T')], inplace=True)
    timeseries.drop(columns=offset_column, inplace=True)
    if feature_column is not None:
        timeseries = timeseries.pivot_table(columns=feature_column, index=timeseries.index)
    # convert index to multi-index with both patients and timedelta stamp
    timeseries.index = pd.MultiIndex.from_tuples(timeseries.index, names=['patient', 'time'])
    return timeseries

def resample_and_mask(timeseries, eICU_path, header, test=False,
                       verbose=False):
    if verbose:
        print('Resampling to 1 hour intervals...')
    # take the mean of any duplicate index entries for unstacking
    timeseries = timeseries.groupby(level=[0, 1]).mean()
    # put patient into columns so that we can round the timedeltas to the nearest hour and take the mean in the time interval
    unstacked = timeseries.unstack(level=0)
    del (timeseries)
    unstacked.index = unstacked.index.ceil(freq='H')
    resampled = unstacked.resample('H', closed='right', label='right').mean()
    del (unstacked)
    mask = resampled.iloc[-24:].notnull()
    mask = mask.astype(int)
    if verbose:
        print('Filling missing data forwards...')
    # carry forward missing values (note they will still be 0 in the nulls table)
    resampled = resampled.fillna(method='ffill').iloc[-24:]
    # simplify the indexes of both tables
    resampled.index = list(range(1, 25))
    mask.index = list(range(1, 25))

    if verbose:
        print('Filling in remaining values with zeros...')
    resampled.fillna(0, inplace=True)

    if verbose:
        print('Reconfiguring and combining features with mask features...')
    # pivot the table around to give the final data
    resampled = resampled.stack(level=1).swaplevel(0, 1).sort_index(level=0)
    mask = mask.stack(level=1).swaplevel(0, 1).sort_index(level=0)

    # rename the columns in pandas for the mask so it doesn't complain
    mask.columns = [str(col) + '_mask' for col in mask.columns]

    # merge the mask with the features
    final = pd.concat([resampled, mask], axis=1)

    if verbose:
        print('Saving progress...')
    # save to csv
    # if test is False:
    final.to_csv(eICU_path + 'preprocessed_timeseries.csv', mode='a', header=header)
    return

def gen_patient_chunk(patients, merged, size=500):
    it = iter(patients)
    chunk = list(itertools.islice(it, size))
    while chunk:
        yield merged.loc[chunk]
        chunk = list(itertools.islice(it, size))

def gen_timeseries_file(eICU_path, test=False):

    print('==> Loading data from timeseries files...')
    timeseries_lab = pd.read_csv(eICU_path + 'timeserieslab.csv')
    timeseries_resp = pd.read_csv(eICU_path + 'timeseriesresp.csv')
    timeseries_periodic = pd.read_csv(eICU_path + 'timeseriesperiodic.csv')
    timeseries_aperiodic = pd.read_csv(eICU_path + 'timeseriesaperiodic.csv')

    print('==> Reconfiguring lab timeseries...')
    timeseries_lab = reconfigure_timeseries(timeseries_lab,
                                            offset_column='labresultoffset',
                                            feature_column='labname',
                                            test=test)
    timeseries_lab.columns = timeseries_lab.columns.droplevel()

    print('==> Reconfiguring respiratory timeseries...')
    # get rid of % signs (found in FiO2 section) and then convert into numbers
    timeseries_resp = timeseries_resp.replace('%', '', regex=True)
    timeseries_resp['respchartnumeric'] = [float(value) for value in timeseries_resp.respchartvalue.values]
    timeseries_resp.drop(columns='respchartvalue', inplace=True)
    timeseries_resp = reconfigure_timeseries(timeseries_resp,
                                             offset_column='respchartoffset',
                                             feature_column='respchartvaluelabel',
                                             test=test)
    timeseries_resp.columns = timeseries_resp.columns.droplevel()

    print('==> Reconfiguring aperiodic timeseries...')
    timeseries_aperiodic = reconfigure_timeseries(timeseries_aperiodic,
                                                  offset_column='observationoffset',
                                                  test=test)

    print('==> Reconfiguring periodic timeseries...')
    timeseries_periodic = reconfigure_timeseries(timeseries_periodic,
                                                 offset_column='observationoffset',
                                                 test=test)

    print('==> Combining data together...')
    merged = timeseries_lab.append(timeseries_resp, sort=False)
    merged = merged.append(timeseries_periodic, sort=False)
    merged = merged.append(timeseries_aperiodic, sort=True)

    print('==> Normalising...')
    # all if not all are not normally distributed
    quantiles = merged.quantile([0.05, 0.95])

    # minus the 'minimum' value and then divide by the 'maximum' value (in a way that is resistant to outliers)
    merged -= quantiles.loc[0.05]
    merged /= quantiles.loc[0.95]

    patients = merged.index.unique(level=0)
    gen_chunks = gen_patient_chunk(patients, merged)
    header = True  # for the first chunk include the header in the csv file
    i = 500
    print('==> Initiating main processing loop...')
    for patient_chunk in gen_chunks:
        resample_and_mask(patient_chunk, eICU_path, header,
                           test=test, verbose=False)
        print('==> Processed ' + str(i) + ' patients...')
        i += 500
        header = False

    return

def add_time_of_day(processed_timeseries, flat_features):

    print('==> Adding time of day features...')
    processed_timeseries = processed_timeseries.join(flat_features[['hour']], how='inner', on='patient')
    processed_timeseries['hour'] = processed_timeseries['time'] + processed_timeseries['hour']
    hour_list = np.linspace(0,1,24)  # make sure it's still scaled well
    processed_timeseries['hour'] = processed_timeseries['hour'].apply(lambda x: hour_list[x - 24])
    return processed_timeseries

def further_processing(eICU_path, test=False):

    processed_timeseries = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
    processed_timeseries.rename(columns={'Unnamed: 1': 'time'}, inplace=True)
    processed_timeseries.set_index('patient', inplace=True)
    flat_features = pd.read_csv(eICU_path + 'flat_features.csv')
    flat_features.rename(columns={'patientunitstayid': 'patient'}, inplace=True)
    flat_features.set_index('patient', inplace=True)

    processed_timeseries = add_time_of_day(processed_timeseries, flat_features)

    if test is False:
        print('==> Saving flat features with non-time varying features added...')
        flat_features.to_csv(eICU_path + 'preprocessed_flat.csv')

        print('==> Saving finalised preprocessed timeseries...')
        # this will replace old one that was updated earlier in the script
        processed_timeseries.to_csv(eICU_path + 'preprocessed_timeseries.csv')

    return

def timeseries_main(eICU_path, test=False):
    # make sure the preprocessed_timeseries.csv file is not there because the first section of this script appends to it
    print('==> Removing the preprocessed_timeseries.csv file if it exists...')
    try:
        os.remove(eICU_path + 'preprocessed_timeseries.csv')

    except FileNotFoundError:
        pass
    gen_timeseries_file(eICU_path, test)
    further_processing(eICU_path, test)
    return

if __name__=='__main__':
    test = False
    eICU_path = './'
    timeseries_main(eICU_path, test)