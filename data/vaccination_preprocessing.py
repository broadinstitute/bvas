"""
Script for pre-processing raw vaccination data.

##################################   REQUIRED OWID DATA   ################################################
# wget https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/vaccinations.csv
# wget https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/us_state_vaccinations.csv
##########################################################################################################
"""
import pandas as pd
import datetime
import numpy as np
import argparse
import sys


def main(args):
    vaccine_status = {'vaccinated': 'people_vaccinated_per_hundred',
                      'fully': 'people_fully_vaccinated_per_hundred'}
    vaccine_status = vaccine_status[args.status]

    try:
        # we use country-level data for all regions except for us states and england/scotland/wales
        data = pd.read_csv('vaccinations.csv')  # country-level data
        usdata = pd.read_csv('us_state_vaccinations.csv')  # us state-level data
    except Exception as e:
        print("Are you sure you have downloaded vaccinations.csv and us_state_vaccinations.csv?\n\n" +
              "wget https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/vaccinations.csv\n" +
              "wget https://github.com/owid/covid-19-data/raw/master/public/data/vaccinations/" +
              "us_state_vaccinations.csv\n")
        print(e)
        sys.exit()

    def date_range(num_days, START_DATE="2019-12-01"):
        start = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
        day = datetime.timedelta(days=1)
        return np.array([start + day * t for t in range(num_days)])

    dates = [date.strftime("%Y-%m-%d") for date in date_range(14 * 56)]
    regions74 = pd.read_csv('74region_summary.csv', index_col=0).Region.values.tolist()
    regions73 = [r for r in regions74 if 'Luxembourg' not in r]  # missing people_fully_vaccinated_per_hundred
    assert len(regions73) == 73

    dfs = []

    for loc in regions73:
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
        df = pd.Series(rates).interpolate(method='linear').fillna(value=0.0)
        df = df.iloc[::14]
        dfs.append(df)

        increments = df.values[1:] - df.values[:-1]
        assert increments.min().item() >= 0.0

    df = pd.concat(dfs, axis=1).transpose()
    df.index = regions73
    df.columns = dates[::14]
    print(df)

    f = 'vaccine_timeseries.{}.csv'.format(vaccine_status)
    df.to_csv(f)
    print("Saving results to {}".format(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse vaccine data')
    parser.add_argument('--status', type=str, default='vaccinated', choices=['vaccinated', 'fully'])
    args = parser.parse_args()

    main(args)
