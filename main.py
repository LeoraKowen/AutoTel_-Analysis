import os

import pandas
import pandas as pd
import geopandas

import matplotlib.pyplot as plt


def plot_data_set(ds, ax):
    ds.plot(x="endLongitude", y="endLatitude", kind="scatter",
            ax=ax, color="red")

    s = ds.sort_values(by='startDate')
    s.plot(x="startLongitude", y="startLatitude", kind="scatter",
           ax=ax, colormap="autumn", c=range(len(ds)))


def _augment(group):
    group.prevActualEnd = group.tripActualEnd.shift()
    group.prevEndLatitude = group.endLatitude.shift()
    group.prevEndLongitude = group.endLongitude.shift()
    return group


def augment_prev_data(df: pandas.DataFrame):
    df = df.sort_values('startDate')
    df['prevActualEnd'] = None
    df['prevEndLatitude'] = None
    df['prevEndLongitude'] = None

    car_groups = df.groupby('carId')
    df = car_groups.apply(_augment)

    df = df.dropna()

    return df


def create_spatial_join(df, long, lat, regions):
    xy = geopandas.points_from_xy(long, lat)
    points = geopandas.GeoDataFrame(df, crs=regions.crs,
                                    geometry=xy)

    points = points[points.is_valid]

    points = geopandas.sjoin(points, regions)

    return pandas.DataFrame(points.drop(columns='geometry'))


def prepare_data():
    dfs = (pd.read_csv(os.path.join('./CSVs', fn),
                       parse_dates=['startDate', 'tripActualEnd'], infer_datetime_format=True) for fn in
           os.listdir('./CSVs'))

    # For fast runs - use only first dataset
    # dfs = list(dfs)[:1]

    df = pd.concat(dfs, ignore_index=True)

    return df


def man_dist(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute Manhattan distance between current start location to previous ride's end location.
    '''
    dist = (df.startLatitude + df.startLongitude) - (df.prevEndLatitude + df.prevEndLongitude)
    return abs(dist)


def get_stationary_intervals(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Filter all intervals to stationary and non-stationary intervals. In stationary intervals, the current drive's start
    location and the previous ride's end location is very small and is non-zero only due to GPS technology inaccuracies.
    In non-stationary intervals (occur due to maintenance or missing entries which occur in big data sets), this
    distance is significant
    '''
    d = man_dist(df)
    return df[d < 0.001]


def canonize_stay_series(s: pd.Series, is_start: int):
    s = s.to_frame('time')
    s['is_stay_start'] = is_start

    return s


def get_stay_intervals(df: pd.DataFrame) -> pd.DataFrame:
    stay_start = canonize_stay_series(df.prevActualEnd, 1)
    stay_end = canonize_stay_series(df.startDate, -1)

    stay_intervals = pd.concat([stay_start, stay_end], ignore_index=True).sort_values(by='time')

    stay_intervals['supply'] = stay_intervals.is_stay_start.cumsum()
    stay_intervals.time = stay_intervals.time.dt.floor('H')
    return stay_intervals


def _supply_per_hour(stay_intervals: pd.DataFrame) -> pd.DataFrame:
    supply = stay_intervals.groupby(by='time').supply.max()
    demand = -stay_intervals[stay_intervals.is_stay_start == -1].groupby(by='time').is_stay_start.sum()

    s_v_d = supply.to_frame()
    s_v_d = s_v_d.join(demand.to_frame('demand'))
    s_v_d = s_v_d.fillna(0)
    s_v_d['s_v_d'] = s_v_d.supply - s_v_d.demand

    return s_v_d


def supply_per_hour(df):
    stay_intervals = get_stay_intervals(df)
    return _supply_per_hour(stay_intervals)


def supply_per_tat_rova(df: pd.DataFrame):
    grouped = df.groupby('ktatrova')

    return grouped.apply(supply_per_hour)


def single_hour_by_month(res: pd.DataFrame, hour):
    filtered_res = res[res.index.get_level_values(1).hour == hour]
    mean_res = filtered_res.groupby(['ktatrova',
                                     filtered_res.index.get_levmel_values(1).day_of_week,
                                     filtered_res.index.get_level_values(1).month,
                                     ]).mean()

    m = mean_res['s_v_d'].unstack()

    return m.groupby(['ktatrova', m.index.get_level_values(1)])


def middle_of_week_by_month(res: pd.DataFrame):
    mean_res = res.groupby(['ktatrova',
                            res.index.get_level_values(1).hour,
                            res.index.get_level_values(1).month,
                            ]).mean()

    m = mean_res['s_v_d'].unstack()

    m = m.fillna(0)

    return m.groupby(['ktatrova'])


def res_by_month(res: pd.DataFrame):
    mean_res = res.groupby(['ktatrova',
                            res.index.get_level_values(1).month,
                            res.index.get_level_values(1).hour,
                            res.index.get_level_values(1).day_of_week,
                            ]).mean()
    mean_res = mean_res.fillna(0)

    m = mean_res['s_v_d'].unstack()

    return m.groupby(['ktatrova', m.index.get_level_values(1)])


def group_results_by_day_of_week(res: pd.DataFrame):
    mean_res = res.groupby(['ktatrova',
                            res.index.get_level_values(1).hour,
                            res.index.get_level_values(1).day_of_week,
                            ]).mean()

    m = mean_res['s_v_d'].unstack()

    return m.groupby('ktatrova')


def plot_groupby_res_single_identifier(r, name, ylim=None):
    p_r = r.plot(legend=None)
    for l in p_r.iteritems():
        if ylim is not None:
            l[1].set_ylim(ylim)

        l[1].get_figure().savefig(f'{name}-{int(l[0])}.jpeg')
        print(l[0])


def heat_map(df, tlv):
    start_points = create_spatial_join(df, df.startLongitude, df.startLatitude, tlv)

    start_size = start_points.groupby('ktatrova').size()

    tlv = tlv.set_index('ktatrova')
    tlv['start_size'] = start_size

    tlv.plot(column='start_size', legend=True)
    plt.show()


if __name__ == "__main__":
    tlv = geopandas.read_file("./export_20220425_220619/Sub Quarters.shp")

    print('Start parsing')
    df = prepare_data()

    print('Adding previous data')
    df = augment_prev_data(df)

    print('Getting stationary intervals')
    stationary_intervals = get_stationary_intervals(df)

    print('spatial')
    stationary_intervals = create_spatial_join(stationary_intervals, stationary_intervals.startLongitude,
                                               stationary_intervals.startLatitude, tlv)

    print('stay intervals')
    res = supply_per_tat_rova(stationary_intervals)

    print('group results')
    grouped = group_results_by_day_of_week(res)

    print('plotting')
    plot_groupby_res_single_identifier(grouped, 'plot')
