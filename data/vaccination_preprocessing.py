"""
Script for pre-processing raw vaccination data.

In particular it can be used to create:
 - vaccine_timeseries.people_fully_vaccinated_per_hundred.csv
 - vaccine_timeseries.people_vaccinated_per_hundred.csv

These files are required if running covid_preprocessing.py with the --phi {vaccinated,fully} argument.

##################################   REQUIRED OWID DATA   #########################################
https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/vaccinations.csv
https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/us_state_vaccinations.csv
###################################################################################################
"""
import argparse
import datetime
import sys

import numpy as np
import pandas as pd


def main(args):
    vaccine_status = {'vaccinated': 'people_vaccinated_per_hundred',
                      'fully': 'people_fully_vaccinated_per_hundred'}
    vaccine_status = vaccine_status[args.status]

    try:
        # we use country-level data for all regions except for us states and england/scotland/wales
        data = pd.read_csv('vaccinations.csv')  # country-level data
        usdata = pd.read_csv('us_state_vaccinations.csv')  # us state-level data
    except Exception as e:
        print("ERROR:\n", e)
        print("Are you sure you have downloaded vaccinations.csv and us_state_vaccinations.csv?\n\n" +
              "wget https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/vaccinations.csv\n" +
              "wget https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/" +
              "us_state_vaccinations.csv\n")
        sys.exit()

    def date_range(num_days, START_DATE="2019-12-01"):
        start = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
        day = datetime.timedelta(days=1)
        return np.array([start + day * t for t in range(num_days)])

    dates = [date.strftime("%Y-%m-%d") for date in date_range(14 * 62)]
    regions = pd.read_csv('128_region_summary.csv', index_col=0).Region.values.tolist()
    regions = [r for r in regions if 'Luxembourg' not in r]  # missing people_fully_vaccinated_per_hundred
    assert len(regions) == 127

    dfs = []

    for loc in regions:
        splits = loc.split(' / ')
        if len(splits) == 3:
            country, region = splits[1:]
        else:
            country = splits[1]

        if country == 'USA':
            if region != 'New York':
                df = usdata[usdata.location == region][['date', vaccine_status]]
            else:
                df = usdata[usdata.location == 'New York State'][['date', vaccine_status]]
        elif country == 'United Kingdom':
            df = data[data.location == region][['date', vaccine_status]]
        else:
            df = data[data.location == country][['date', vaccine_status]]

        assert len(df) > 0

        rates = []
        for date in dates:
            if vaccine_status == 'people_fully_vaccinated_per_hundred':
                rate = df[df.date == date].people_fully_vaccinated_per_hundred.values
            elif vaccine_status == 'people_vaccinated_per_hundred':
                rate = df[df.date == date].people_vaccinated_per_hundred.values
            rate = rate.item() if rate.shape == (1,) else np.nan
            rates.append(rate / 100.0)
        # linear interpolation
        df = pd.Series(rates).interpolate(method='linear').fillna(value=0.0)
        df = df.iloc[::14]
        dfs.append(df)

        increments = df.values[1:] - df.values[:-1]
        assert increments.min().item() >= 0.0

    df = pd.concat(dfs, axis=1).transpose()
    df.index = regions
    df.columns = dates[::14]
    print(df)

    f = 'vaccine_timeseries.{}.csv'.format(vaccine_status)
    df.to_csv(f)
    print("Saving results to {}".format(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse vaccine data')
    parser.add_argument('--status', type=str, default='fully', choices=['vaccinated', 'fully'])
    args = parser.parse_args()

    main(args)
