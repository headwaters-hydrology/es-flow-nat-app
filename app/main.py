"""

"""
import io
import os
import geopandas as gpd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import urllib
from tethysts import utils
import requests
import zstandard as zstd
import orjson
import flask
import codecs
import pickle
import shapely
import yaml
from gistools import vector
import pyproj
# import dash_leaflet as dl
# import dash_leaflet.express as dlx
import copy
import xarray as xr
from flask_caching import Cache
import base64
from tethysts import Tethys
# from dash_extensions.javascript import Namespace
# from util import app_ts_summ, sel_ts_summ, ecan_ts_data

pd.options.display.max_columns = 10

# external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']

# server = flask.Flask(__name__)
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server,  url_base_pathname = '/')
# app = dash.Dash(__name__, server=server,  url_base_pathname = '/')

# app = dash.Dash(__name__)
# server = app.server

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server,  url_base_pathname = '/')

##########################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)

remotes = param['remotes']

flow_remote = [r for r in remotes if r['bucket'] == 'es-hilltop'][0]

ts_plot_height = 600
map_height = 700

lat1 = -45.74
lon1 = 168.14
zoom1 = 7

mapbox_access_token = "pk.eyJ1IjoibXVsbGVua2FtcDEiLCJhIjoiY2pudXE0bXlmMDc3cTNxbnZ0em4xN2M1ZCJ9.sIOtya_qe9RwkYXj5Du1yg"

collection_methods = ['Gauging', 'Recorder']
# collection_methods = {'Gauging': '74c5bcd07846abae0e28ddd2', 'Recorder': '4ae05d099af292fec48792ec'}

# base_url = 'http://tethys-api-int:80/tethys/data/'
# base_url = 'https://api-int.tethys-ts.xyz/tethys/data/'

# catch_key_base = 'tethys/station_misc/{station_id}/catchment.geojson.zst'

summ_table_cols = ['Station reference', 'Min', 'Q95', 'Median', 'Q5', 'Max', 'Start Date', 'End Date', 'Catchment Area (ha)']
reg_table_cols = ['NRMSE', 'MANE', 'Adj R2', 'Number of observations', 'Correlated sites', 'F value']

cache_config = {
    # "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "FileSystemCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 24*60*60,
    'CACHE_DIR': param['cache_path']
}

cache = Cache(server, config=cache_config)

tabs_styles = {
    'height': '40px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '5px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '5px'
}

mat_stns_json = 'mataura_protected_waters_stns.json'

with open(os.path.join(base_dir, mat_stns_json), 'r') as infile:
    mat_stns = orjson.loads(infile.read())

catch_zstd = 'es_flow_sites_catchment_delin_2021-10-06.pkl.zstd'

with open(os.path.join(base_dir, catch_zstd), 'rb') as infile:
    rec_shed = utils.read_pkl_zstd(infile.read(), True)

###############################################
### Functions


def encode_obj(obj):
    """

    """
    cctx = zstd.ZstdCompressor(level=1)
    c_obj = codecs.encode(cctx.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)), encoding="base64").decode()


    return c_obj


def decode_obj(str_obj):
    """

    """
    dctx = zstd.ZstdDecompressor()
    obj1 = dctx.decompress(codecs.decode(str_obj.encode(), encoding="base64"))
    d1 = pickle.loads(obj1)

    return d1


def build_reg_table(site_summ):
    """

    """
    table1 = [{'NRMSE': s['nrmse'], 'MANE': s['mane'], 'Adj R2': s['Adj R2'], 'Number of observations': s['nobs'], 'Correlated sites': s['x sites'], 'F value': s['f value']} for i, s in site_summ.iterrows()]

    return table1


# def build_stats(results):
#     """

#     """
#     r1 = results['streamflow']

#     ref = str(results['ref'].values)

#     min1 = r1.min().values.round(3)
#     max1 = r1.max().values.round(3)
#     median1 = r1.median().values.round(3)
#     q5 = r1.quantile(0.95).values.round(3)
#     q95 = r1.quantile(0.05).values.round(3)
#     count1 = int(r1.count().values)
#     from_date = (pd.Timestamp(results.time.min().values).round('D') + pd.DateOffset(hours=12)).strftime('%Y-%m-%d')
#     to_date = (pd.Timestamp(results.time.max().values).round('D') + pd.DateOffset(hours=12)).strftime('%Y-%m-%d')

#     stats_dict = {'Station reference': ref, 'Min': min1, 'Q95': q95, 'Median': median1, 'Q5': q5, 'Max': max1, 'Count': count1, 'Start Date': from_date, 'End Date': to_date}

#     return stats_dict


def build_summ_table(results):
    """

    """
    r1 = results['streamflow']

    ref = str(results['ref'].values)

    min1 = r1.min().values.round(3)
    max1 = r1.max().values.round(3)
    median1 = r1.median().values.round(3)
    q5 = r1.quantile(0.95).values.round(3)
    q95 = r1.quantile(0.05).values.round(3)
    count1 = int(r1.count().values)
    from_date = (pd.Timestamp(results.time.min().values).round('D') + pd.DateOffset(hours=12)).strftime('%Y-%m-%d')
    to_date = (pd.Timestamp(results.time.max().values).round('D') + pd.DateOffset(hours=12)).strftime('%Y-%m-%d')

    area1 = int(round(rec_shed[rec_shed.station_id == results['station_id'].values].iloc[0]['area']/10000))

    table1 = {'Station reference': ref, 'Min': min1, 'Q95': q95, 'Median': median1, 'Q5': q5, 'Max': max1, 'Count': count1, 'Start Date': from_date, 'End Date': to_date, 'Catchment Area (ha)': area1}

    return table1


def get_stations(tethys, dataset_id):
    """

    """
    fn_stns = tethys.get_stations(dataset_id)

    return fn_stns


def get_results(tethys, dataset_id, station_id, from_date=None, to_date=None):
    """

    """
    data2 = tethys.get_results(dataset_id, station_id, from_date=None, to_date=None, squeeze_dims=True, output='Dataset')
    data3 = data2
    data3['time'] = pd.to_datetime(data3['time'].values) + pd.DateOffset(hours=12)
    coords = list(data3.coords)
    if 'geometry' in coords:
        data3 = data3.drop('geometry')
    if 'height' in coords:
        data3 = data3.drop('height')

    return data3


# def get_catchment(station_id, remote):
#     """

#     """
#     bucket = remote['bucket']

#     key1 = catch_key_base.format(station_id=station_id)
#     obj = utils.get_object_s3(key1, remote['connection_config'], bucket, 'zstd')
#     b2 = io.BytesIO(obj)
#     c1 = gpd.read_file(b2)

#     return c1


def get_catchment(station_id):
    """

    """
    c1 = rec_shed[rec_shed.station_id == station_id]

    return c1


def stns_dict_to_gdf(stns):
    """

    """
    stns1 = copy.deepcopy(stns)
    geo1 = [shapely.geometry.Point(s['geometry']['coordinates']) for s in stns1]

    [s.update({'min': s['stats']['min'], 'max': s['stats']['max'], 'median': s['stats']['median'], 'from_date': s['time_range']['from_date'], 'to_date': s['time_range']['to_date']}) for s in stns1]
    [(s.pop('stats'), s.pop('geometry'), s.pop('time_range')) for s in stns1]

    df1 = pd.DataFrame(stns1)
    df1['from_date'] = pd.to_datetime(df1['from_date'])
    df1['to_date'] = pd.to_datetime(df1['to_date'])
    df1['modified_date'] = pd.to_datetime(df1['modified_date'])

    stns_gpd1 = gpd.GeoDataFrame(df1, crs=4326, geometry=geo1)

    return stns_gpd1


def combine_flows(flow_meas, flow_nat):
    """

    """
    stn_ref = str(flow_meas['ref'].values)

    meas_data1 = flow_meas['streamflow']
    nat_data1 = flow_nat['naturalised_streamflow']

    ## Combine
    flow1 = xr.merge([meas_data1, nat_data1], compat='override').to_dataframe()
    flow1.rename(columns={'streamflow': 'flow', 'naturalised_streamflow': 'nat flow'}, inplace=True)

    return flow1, stn_ref


@cache.memoize()
def get_flow_duration(flow_meas, flow_nat):
    """

    """
    ## Get data
    flow1, stn_ref = combine_flows(flow_meas, flow_nat)

    ## Measured flow
    flow0 = flow1[['flow']].dropna().copy()

    flow0['rank_flow'] = flow0['flow'].rank(numeric_only=True, ascending=False).astype(int)
    flow0['rank_percent_flow'] = (flow0['rank_flow'] / flow0['rank_flow'].max()) * 100

    flow2 = flow0.sort_values('rank_flow')

    ## Nat flow
    nat_flow = flow1[['nat flow']].dropna().copy()

    nat_flow['rank_nat_flow'] = nat_flow['nat flow'].rank(numeric_only=True, ascending=False).astype(int)
    nat_flow['rank_percent_nat_flow'] = (nat_flow['rank_nat_flow'] / nat_flow['rank_nat_flow'].max()) * 100

    flow3 = nat_flow.sort_values('rank_nat_flow')

    return flow2, flow3, stn_ref


@cache.memoize()
def get_flow_allo(allo, use, flow_meas):
    """

    """
    stn_ref = str(allo['ref'].values)

    ## Get data
    allo1 = allo['allocation']
    use1 = use['water_use']

    # Meas flow Q95
    flow_q95 = flow_meas['streamflow'].quantile(0.05).values

    ## Combine
    # allo_use1 = pd.concat([flow1, allo1, use1], axis=1).dropna()
    allo_use1 = xr.merge([allo1, use1, flow_meas['streamflow']], compat='override').to_dataframe().dropna()

    return allo_use1, stn_ref, flow_q95


@cache.memoize()
def get_cumulative_flows(flow_meas):
    """

    """
    today1 = pd.Timestamp.now()

    ## Get data
    stn_ref = str(flow_meas['ref'].values)

    meas_data1 = flow_meas['streamflow']

    flow4 = meas_data1.to_dataframe().dropna()

    ## Process data
    flow4.rename(columns={'streamflow': 'flow'}, inplace=True)

    flow4[['flow']] = flow4[['flow']] * 60*60*24*0.000000001

    cumsum1 = flow4.groupby(pd.Grouper(level='time', freq='A-JUN')).cumsum()

    cumsum1['day'] = cumsum1.index.dayofyear
    cumsum1['month'] = cumsum1.index.month
    cumsum1['year'] = cumsum1.index.year

    base_year = np.where(cumsum1['month'].ge(7), cumsum1['year'], cumsum1['year'].sub(1))
    cumsum1['water_year'] = base_year
    base_date = pd.to_datetime(base_year, format='%Y') + pd.DateOffset(months=6)
    cumsum1['day'] = (cumsum1.index - base_date).days + 1
    if today1.month > 7:
        today_wateryear = today1.year
    else:
        today_wateryear = today1.year - 1
    today_dayofyear = (today1 - base_date[-1]).days + 1

    cumsum2 = cumsum1.loc[cumsum1.day < 366, ['water_year', 'day', 'flow']].copy()

    current_cumsum1 = cumsum2[cumsum2['water_year'] == today_wateryear].set_index('day')[['flow']].copy()

    day_count = cumsum2.groupby('water_year')['day'].count()
    good_years = day_count[day_count >= 350].index

    if len(good_years) < 4:
        return None

    else:
        cumsum3 = cumsum2[cumsum2['water_year'].isin(good_years)]
        max_flow = cumsum3.groupby('water_year')['flow'].max()

        median_year = max_flow[max_flow == max_flow.quantile(0.5, 'nearest')].index[0]
        q95_year = max_flow[max_flow == max_flow.quantile(0.95, 'nearest')].index[0]
        q5_year = max_flow[max_flow == max_flow.quantile(0.05, 'nearest')].index[0]

        median_day = cumsum3.loc[cumsum3['water_year'] == median_year, ['day', 'flow']].set_index('day')

        q95_day = cumsum3.loc[cumsum3['water_year'] == q95_year, ['day', 'flow']].set_index('day')
        q95_day_rev = q95_day[::-1]

        q5_day = cumsum3.loc[cumsum3['water_year'] == q5_year, ['day', 'flow']].set_index('day')
        q5_day_rev = q5_day[::-1]

        x = q5_day.index.tolist()
        x_rev = x[::-1]

        return x, x_rev, median_day, q95_day, q95_day_rev, q5_day, q5_day_rev, stn_ref, median_year, q95_year, q5_year, current_cumsum1, today_dayofyear


def fig_flow_duration(flow_meas, flow_nat):
    """

    """
    colors1 = ['rgb(102,194,165)', 'red', 'rgb(252,141,0)', 'rgb(141,160,203)']

    print('fig_fd triggered')

    ## Get data
    flow2, flow3, stn_ref = get_flow_duration(flow_meas, flow_nat)

    orig = go.Scattergl(
        x=flow2['rank_percent_flow'],
        y=flow2['flow'],
        name = 'Measured Flow',
        line = dict(width=5, color = colors1[3]),
        opacity = 0.8)

    q95_trace = go.Scattergl(
        x=[95],
        y=[flow2['flow'].quantile(0.05)],
        name = 'Q95 Measured Flow',
        mode='markers',
        marker_color='blue',
        marker_size=20,
        # line = dict(width=10,color = colors1[1]),
        opacity = 1)
    nat = go.Scattergl(
        x=flow3['rank_percent_nat_flow'],
        y=flow3['nat flow'],
        name = 'Naturalised Flow',
        line = dict(width=5, color = colors1[0]),
        opacity = 0.8)

    q95_nat_trace = go.Scattergl(
        x=[95],
        y=[flow3['nat flow'].quantile(0.05)],
        name = 'Q95 Naturalised Flow',
        mode='markers',
        marker_color='red',
        marker_size=12,
        # line = dict(width=10,color = colors1[1]),
        opacity = 1)

    data = [nat, orig, q95_trace, q95_nat_trace]

    layout = dict(
        title=stn_ref,
        yaxis={'title': 'Flow rate (m3/s)'},
        xaxis={'title': 'Percent of time flow is exceeded'},
        # dragmode='pan',
        font=dict(size=18),
        hovermode='x',
        paper_bgcolor = '#F4F4F8',
        plot_bgcolor = '#F4F4F8',
        height = ts_plot_height
        )

    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(type="log")
    fig.update_xaxes(dtick=10)

    return fig


def fig_cumulative_flow(flow_nat):
    """

    """
    results = get_cumulative_flows(flow_nat)

    if results is None:
        return 'Not enough years for statistics'
    else:
        x, x_rev, median_day, q95_day, q95_day_rev, q5_day, q5_day_rev, stn_ref, median_year, q95_year, q5_year, current_cumsum1, today_dayofyear = results

        layout = dict(
            title=stn_ref,
            yaxis={'title': 'Flow volume (km3)'},
            xaxis={'title': 'Day of year starting on July 1st'},
            # dragmode='pan',
            font=dict(size=18),
            hovermode='x',
            paper_bgcolor = '#F4F4F8',
            plot_bgcolor = '#F4F4F8',
            height = ts_plot_height
            )

        fig = go.Figure(layout=layout)

        fig.add_vline(x=today_dayofyear,
                      line_width=3,
                      line_dash="dash",
                      line_color="green",
                      annotation_text='Today',
                      opacity=0.5,
                      annotation_position='top')

        fig.add_trace(go.Scattergl(
            x=x+x_rev,
            y=q95_day['flow'].tolist()+q5_day_rev['flow'].tolist(),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name='95th percentile range',
            opacity = 0.5
            ))

        fig.add_trace(go.Scattergl(
            x=x, y=q95_day['flow'].tolist(),
            line_color='rgb(73,56,41)',
            name='Q5 Flow (' + str(q95_year) + ')',
            line_width=2
            ))

        fig.add_trace(go.Scattergl(
            x=x, y=q5_day['flow'].tolist(),
            line_color='rgb(143,59,27)',
            name='Q95 Flow (' + str(q5_year) + ')',
            line_width=2
            ))

        fig.add_trace(go.Scattergl(
            x=x, y=median_day['flow'].tolist(),
            line_color='rgb(141,160,203)',
            name='Median flow (' + str(median_year) + ')',
            line_width=5
            ))

        if not current_cumsum1.empty:
            fig.add_trace(go.Scattergl(
                x=current_cumsum1.index, y=current_cumsum1['flow'].tolist(),
                line_color='rgb(0,160,240)',
                name="This year's flow",
                line_width=3
                ))

        fig.update_traces(mode='lines')

        return fig


def fig_hydrograph(flow_meas, flow_nat):
    """

    """
    colors1 = ['rgb(102,194,165)', 'red', 'rgb(252,141,0)', 'rgb(141,160,203)']

    ## Get data
    flow1, stn_ref = combine_flows(flow_meas, flow_nat)
    flow2 = flow1.resample('D').first().interpolate('linear', limit=7)

    orig = go.Scattergl(
        x=flow2.index,
        y=flow2['flow'],
        name = 'Measured Flow',
        line = dict(width=2, color = colors1[3]),
        opacity = 0.8)

    nat = go.Scattergl(
        x=flow2.index,
        y=flow2['nat flow'],
        name = 'Naturalised Flow',
        line = dict(width=2, color = colors1[0]),
        opacity = 0.8)

    data = [nat, orig]

    layout = dict(
        title=stn_ref,
        yaxis={'title': 'Flow rate (m3/s)'},
        xaxis={'title': 'Date'},
        # dragmode='pan',
        font=dict(size=18),
        hovermode='x',
        paper_bgcolor = '#F4F4F8',
        plot_bgcolor = '#F4F4F8',
        height = ts_plot_height
        )

    fig = go.Figure(data=data, layout=layout)
    # fig.update_yaxes(type="log")
    # fig.update_xaxes(dtick=10)

    return fig


def fig_allo_use(allo, use, flow_meas, stn_id):
    """

    """
    colors1 = ['rgb(102,194,165)', 'red', 'rgb(252,141,0)', 'rgb(141,160,203)']

    ## Get data
    allo_use1, stn_ref, flow_q95 = get_flow_allo(allo, use, flow_meas)

    allo = go.Scattergl(
        x=allo_use1.index,
        y=allo_use1['allocation'],
        name = 'Allocation',
        line = dict(width=2, color = colors1[3]),
        opacity = 0.8)

    use = go.Scattergl(
        x=allo_use1.index,
        y=allo_use1['water_use'],
        name = 'Abstraction',
        line = dict(width=2, color = colors1[0]),
        opacity = 0.8)

    if stn_id in mat_stns:
        y1 = allo_use1['streamflow'] * 0.05
        y1.loc[y1 > allo_use1['allocation']] = allo_use1['allocation'][y1 > allo_use1['allocation']]
        text1 = '5% of daily mean flow'
    else:
        y1 = [flow_q95*0.3] * len(allo_use1.index)
        text1 = '30% of Q95 flow'

    q95 = go.Scattergl(
        x=allo_use1.index,
        y=y1,
        name = text1,
        line = dict(width=2, color = colors1[2]),
        opacity = 0.8)

    data = [allo, use, q95]

    layout = dict(
        title=stn_ref,
        yaxis={'title': 'Flow rate (m3/s)'},
        xaxis={'title': 'Date'},
        # dragmode='pan',
        font=dict(size=18),
        hovermode='x',
        paper_bgcolor = '#F4F4F8',
        plot_bgcolor = '#F4F4F8',
        height = ts_plot_height
        )

    fig = go.Figure(data=data, layout=layout)
    # fig.update_yaxes(type="log")
    # fig.update_xaxes(dtick=10)

    return fig


def plot_catch_map(stn_dict, c_method, active, layout, wap_stn_data=None, flow_stn_id=None):
    """

    """
    ### Figure
    fig = go.Figure(layout=layout)

    ### Flow stations
    flow_stn_data1 = stn_dict[active]['naturalised'][c_method]

    if isinstance(flow_stn_id, str):

        ### Catchments
        catch1 = get_catchment(flow_stn_id)
        catch1.crs = pyproj.CRS.from_epsg(2193)
        feature = catch1.to_crs(4326).iloc[0]['geometry']

        catch_lons = []
        catch_lats = []

        if isinstance(feature, shapely.geometry.polygon.Polygon):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multipolygon.MultiPolygon):
            linestrings = feature.geoms

        for linestring in linestrings:
            x, y = linestring.exterior.xy

            catch_lons.extend(x.tolist())
            catch_lats.extend(y.tolist())

        # catch_lons.extend([None])
        # catch_lats.extend([None])

        ### Waps
        waps1 = vector.sel_sites_poly(wap_stn_data.to_crs(2193), catch1).to_crs(4326)
        # print(waps1)

        ### Add to figure
        fig.add_trace(go.Scattermapbox(
            fill = "toself",
            lon = catch_lons,
            lat = catch_lats,
            mode='lines',
            hoverinfo='none',
            fillcolor='rgba(0, 0, 200, 0.1)',
            line_color='rgba(225, 225, 255, 0.5)',
            name='catchment'
            ))

        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = waps1['geometry'].x,
            lat = waps1['geometry'].y,
            text=waps1['ref'],
            marker_color='black',
            marker_size=6,
            name='abstraction sites',
            hoverinfo='text'
            ))

        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = flow_stn_data1.geometry.x,
            lat = flow_stn_data1.geometry.y,
            hovertext=flow_stn_data1['ref'],
            # text=flow_stn_names,
            ids=flow_stn_data1['station_id'],
            marker_color='rgba(200, 0, 0, 0.3)',
            marker_size=10,
            name='flow sites',
            textposition='top center',
            hoverinfo='text'
            ))

    else:

        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lon = flow_stn_data1.geometry.x,
            lat = flow_stn_data1.geometry.y,
            hovertext=flow_stn_data1['ref'],
            # text=flow_stn_data1['ref'],
            ids=flow_stn_data1['station_id'],
            marker_color='red',
            marker_size=10,
            name='flow sites',
            textposition='top center',
            hoverinfo='text'
            ))

    return fig


def plot_flow_stns(stn_dict, c_method, active, layout):
    """

    """
    flow_stn_data1 = stn_dict[active]['naturalised'][c_method]

    flow_stn_trace = go.Scattermapbox(
        mode = "markers",
        lon = flow_stn_data1.geometry.x,
        lat = flow_stn_data1.geometry.y,
        hovertext=flow_stn_data1['ref'],
        # text=flow_stn_data1['ref'],
        ids=flow_stn_data1['station_id'],
        marker_color='red',
        marker_size=10,
        name='flow sites',
        textposition='top center',
        hoverinfo='text'
        )

    fig = dict(data=[flow_stn_trace], layout=layout)

    return fig


def get_flow_stn_list(stn_dict, c_method, active):
    """

    """
    f1 = stn_dict[active]['naturalised'][c_method]

    sites2 = [{'label': s['ref'], 'value': s['station_id']} for s in f1.iterrows()]

    return sites2


############################################
### The app
map_layout = dict(mapbox = dict(layers = [], accesstoken = mapbox_access_token, style = "outdoors", center=dict(lat=lat1, lon=lon1), zoom=zoom1), margin = dict(r=0, l=0, t=0, b=0), autosize=True, hovermode='closest', height=map_height, showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

# @server.route('/wai-vis')
# def main():
def serve_layout():
    ################################################
    ### Initialize base data
    run_date = pd.Timestamp.now(tz='utc').round('s').tz_localize(None)
    last_month = (run_date - pd.tseries.offsets.MonthEnd(1)).floor('D')
    last_year = ((last_month - pd.DateOffset(years=1) - pd.DateOffset(days=2)) + pd.tseries.offsets.MonthEnd(1)) + pd.DateOffset(days=1)

    ## Get datasets and filter
    tethys = Tethys(remotes)

    datasets = tethys.datasets.copy()

    fn_ds = [ds for ds in datasets if ds['parameter'] == 'naturalised_streamflow'][0]

    rec_flow_ds = [ds for ds in datasets if (ds['parameter'] == 'streamflow') and (ds['method'] == 'sensor_recording') and (ds['product_code'] == 'quality_controlled_data') and (ds['owner'] == 'Environment Southland') and (ds['frequency_interval'] =='24H')][0]

    man_flow_ds = [ds for ds in datasets if (ds['parameter'] == 'streamflow') and (ds['method'] == 'simulation') and (ds['product_code'] == 'log-log linear regression') and (ds['owner'] == 'Environment Southland') and (ds['frequency_interval'] =='24H')][0]

    use_ds = [ds for ds in datasets if (ds['parameter'] == 'water_use') and (ds['method'] == 'simulation') and (ds['product_code'] == 'estimation method 1') and (ds['owner'] == 'Environment Southland') and (ds['frequency_interval'] =='24H')][0]

    flow_use_ds = [ds for ds in datasets if (ds['parameter'] == 'water_use') and (ds['method'] == 'simulation') and (ds['product_code'] == 'stream depletion method 1') and (ds['owner'] == 'Environment Southland') and (ds['frequency_interval'] =='24H')][0]

    allo_ds = [ds for ds in datasets if (ds['parameter'] == 'allocation') and (ds['method'] == 'simulation') and (ds['product_code'] == 'stream depletion allocation method 1') and (ds['owner'] == 'Environment Southland') and (ds['frequency_interval'] =='24H')][0]

    allo_ds_id = allo_ds['dataset_id']
    flow_use_ds_id = flow_use_ds['dataset_id']

    ds_ids_dict =   {'measured':
                        {'Recorder': rec_flow_ds['dataset_id'],
                         'Gauging': man_flow_ds['dataset_id']
                         },
                    'naturalised':
                        {'Recorder': fn_ds['dataset_id'],
                         'Gauging': fn_ds['dataset_id']
                         },
                    'allocation': allo_ds_id,
                    'abstraction': flow_use_ds_id
                    }


    ## Get stations
    # rec flow
    rec_stns = get_stations(tethys, rec_flow_ds['dataset_id'])
    rec_stns_active = [s for s in rec_stns if pd.Timestamp(s['time_range']['to_date']) > last_year]
    rec_stns_ids = set([s['station_id'] for s in rec_stns])
    rec_stns_active_ids = set([s['station_id'] for s in rec_stns_active])

    # man flow
    man_stns = get_stations(tethys, man_flow_ds['dataset_id'])
    man_stns = [s for s in man_stns if not s['station_id'] in rec_stns_ids]
    man_stns_active = [s for s in man_stns if pd.Timestamp(s['time_range']['to_date']) > last_year]
    man_stns_ids = set([s['station_id'] for s in man_stns])
    man_stns_active_ids = set([s['station_id'] for s in man_stns_active])

    # Flow nat
    fn_stns = get_stations(tethys, fn_ds['dataset_id'])

    rec_fn_stns = [s for s in fn_stns if s['station_id'] in rec_stns_ids]
    man_fn_stns = [s for s in fn_stns if s['station_id'] in man_stns_ids]

    # Process regression data
    for s in man_fn_stns:
        if 'properties' in s:
            for r in s['properties']:
                if r == 'x sites':
                    x_sites1 = s['properties'][r]['data'].split(', ')

                    refs = []
                    for x in x_sites1:
                        ref = [rec['ref'] for rec in rec_fn_stns if rec['station_id'] == x]
                        refs.extend(ref)

                    s[r] = ', '.join(refs)

                else:
                    s[r] = s['properties'][r]['data']
            s.pop('properties')

    rec_fn_stns_active = [s for s in fn_stns if s['station_id'] in rec_stns_active_ids]
    man_fn_stns_active = [s for s in man_fn_stns if s['station_id'] in man_stns_active_ids]

    stn_name_dict = {'all': {'Recorder': [{'label': s['ref'], 'value': s['station_id']} for s in rec_fn_stns], 'Gauging':[{'label': s['ref'], 'value': s['station_id']} for s in man_fn_stns]},
                     'active': {'Recorder': [{'label': s['ref'], 'value': s['station_id']} for s in rec_fn_stns_active], 'Gauging':[{'label': s['ref'], 'value': s['station_id']} for s in man_fn_stns_active]}}

    ## Create stn dicts
    stn_dict = {'all':
                {'measured': {'Recorder': stns_dict_to_gdf(rec_stns), 'Gauging': stns_dict_to_gdf(man_stns)},
                 'naturalised': {'Recorder': stns_dict_to_gdf(rec_fn_stns), 'Gauging': stns_dict_to_gdf(man_fn_stns)}
                 },
                'active':
                    {'measured': {'Recorder': stns_dict_to_gdf(rec_stns_active), 'Gauging': stns_dict_to_gdf(man_stns_active)},
                 'naturalised': {'Recorder': stns_dict_to_gdf(rec_fn_stns_active), 'Gauging': stns_dict_to_gdf(man_fn_stns_active)}}}

    # water use
    wu_stns = get_stations(tethys, use_ds['dataset_id'])
    wap_stn_data = stns_dict_to_gdf(wu_stns)

    # Allocation
    # allo_stns = get_stations(base_url, allo_ds['dataset_id'])
    # allo_stn_data = stns_dict_to_gdf(allo_stns)

    ## Methods
    c_methods = collection_methods.copy()

    c_method_init = 'Recorder'

    # dataset_table_cols = {'license': 'Data License', 'precision': 'Data Precision', 'units': 'Units'}

    ### prepare summaries and initial states
    init_sites = stn_name_dict['active'][c_method_init]

    fig_mp = plot_flow_stns(stn_dict, c_method_init, 'active', map_layout)

    layout = html.Div(children=[
    html.Div([
        html.P(children='Select dataset:'),
        html.Label('Collection method'),
        dcc.Dropdown(options=[{'label': d, 'value': d} for d in c_methods], value=c_method_init, id='method_dd', clearable=False),
        html.Label('Site name'),
        dcc.Dropdown(options=init_sites, id='sites', optionHeight=40),
        dcc.RadioItems(id='active_select',
            options=[
                {'label': 'Active flow sites', 'value': 'active'},
                {'label': 'All flow sites', 'value': 'all'}
            ],
            value='active'),
        dcc.Markdown('''
            *Active flow sites only include sites that have data within the last year.
            '''),
        dcc.Link(html.Img(src=app.get_asset_url('es-logo.svg')), href='https://www.es.govt.nz/')
        ],
    className='two columns', style={'margin': 20}),

    html.Div([
        html.P('Click on a site:', style={'display': 'inline-block'}),
        dcc.Graph(
            id = 'site-map',
            style={'height': map_height},
            figure=fig_mp,
            config={"displaylogo": False})


    ], className='three columns', style={'margin': 20}),
#
    html.Div([

        dcc.Loading(
                id="loading-plots",
                type="default",
                children=[dcc.Tabs(id='plot_tabs', value='info_tab', style=tabs_styles, children=[
                            dcc.Tab(label='Info', value='info_tab', style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(label='Flow duration', value='fd_plot', style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(label='Cumulative flow', value='cf_plot', style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(label='Hydrograph', value='hydro_plot', style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(label='Allocation', value='allo_plot', style=tab_style, selected_style=tab_selected_style),
                            ]
                        ),
        html.Div(id='plots')
        ]),

    dash_table.DataTable(
        id='summ_table',
        columns=[{"name": v, "id": v, 'deletable': True} for v in summ_table_cols],
        data=[],
        sort_action="native",
        sort_mode="multi",
        style_cell={
            'minWidth': '80px', 'maxWidth': '200px',
            'whiteSpace': 'normal'
        }
        ),
    dash_table.DataTable(
        id='reg_table',
        columns=[{"name": v, "id": v, 'deletable': True} for v in reg_table_cols],
        data=[],
        sort_action="native",
        sort_mode="multi",
        style_cell={
            'minWidth': '80px', 'maxWidth': '200px',
            'whiteSpace': 'normal'
        }
        )

    ], className='six columns', style={'margin': 10, 'height': 900}),
    dcc.Store(id='tethys', data=encode_obj(tethys)),
    dcc.Store(id='flow_meas', data=''),
    dcc.Store(id='flow_nat', data=''),
    dcc.Store(id='allocation', data=''),
    dcc.Store(id='abstraction', data=''),
    dcc.Store(id='stn_dict', data=encode_obj(stn_dict)),
    dcc.Store(id='wap_stn_data', data=encode_obj(wap_stn_data)),
    dcc.Store(id='stn_names', data=orjson.dumps(stn_name_dict).decode()),
    dcc.Store(id='ds_ids', data=orjson.dumps(ds_ids_dict).decode()),
    dcc.Store(id='run_date', data=str(run_date)),
    dcc.Store(id='last_month', data=str(last_month)),
    dcc.Store(id='last_year', data=str(last_year)),
], style={'margin':0})

    return layout


app.layout = serve_layout

#########################################
### Callbacks


@app.callback(
        Output('sites', 'value'),
        [Input('site-map', 'selectedData'), Input('site-map', 'clickData')]
        )
def update_sites_values(selectedData, clickData):
    # print(clickData)
    # print(selectedData)
    if selectedData:
        sel1 = selectedData['points'][0]
        if 'id' in sel1:
            site1_id = sel1['id']
        else:
            site1_id = None
    elif clickData:
        sel1 = clickData['points'][0]
        if 'id' in sel1:
            site1_id = sel1['id']
        else:
            site1_id = None
    else:
        site1_id = None

    # print(sites1_id)

    return site1_id


@app.callback(
    Output('summ_table', 'data'),
    [Input('flow_meas', 'data')])
@cache.memoize()
def update_summ_table(flow_meas_str):
    if flow_meas_str is not None:
        if len(flow_meas_str) > 1:
            flow_meas = decode_obj(flow_meas_str)
            summ_table = build_summ_table(flow_meas)

            return [summ_table]
        else:
            return []
    else:
        return []


@app.callback(
    Output('reg_table', 'data'),
    [Input('sites', 'value')],
    [State('stn_dict', 'data'), State('method_dd', 'value'), State('active_select', 'value')])
@cache.memoize()
def update_reg_table(site, stn_dict_str, c_method, active):
    if site is not None:
        if c_method == 'Gauging':
            stn_dict = decode_obj(stn_dict_str)
            meas_flow_stn_data1 = stn_dict[active]['naturalised'][c_method]
            meas1 = meas_flow_stn_data1[meas_flow_stn_data1['station_id'] == site]
            # print(meas1)

            if not meas1.empty:

                summ_table = build_reg_table(meas1)

                return summ_table
            else:
                return []
        else:
            return []
    else:
        return []


@app.callback(
        Output('sites', 'options'),
        [Input('method_dd', 'value'), Input('active_select', 'value')],
        [State('stn_names', 'data')])
@cache.memoize()
def update_sites_options(c_method, active, stn_names_json):
    stn_names_dict = orjson.loads(stn_names_json)

    stn_options = stn_names_dict[active][c_method]

    return stn_options


@app.callback(Output('site-map', 'figure'),
              [Input('method_dd', 'value'), Input('sites', 'value'), Input('active_select', 'value')],
              [State('stn_dict', 'data'), State('wap_stn_data', 'data'), State('site-map', 'figure')])
@cache.memoize()
def render_map_complex(c_method, flow_stn_id, active, stn_dict_str, wap_stn_data_str, old_fig):
    # print(flow_stn_id)

    stn_dict = decode_obj(stn_dict_str)
    map_layout = old_fig['layout']

    if isinstance(flow_stn_id, str):
        wap_stn_data = decode_obj(wap_stn_data_str)

        fig_mp = plot_catch_map(stn_dict, c_method, active, map_layout, wap_stn_data, flow_stn_id)
    else:
        fig_mp = plot_catch_map(stn_dict, c_method, active, map_layout)

    return fig_mp


@app.callback(
    Output('flow_meas', 'data'),
    [Input('sites', 'value')],
    [State('tethys', 'data'), State('ds_ids', 'data'), State('method_dd', 'value')])
# @cache.memoize()
def get_flow_meas_data(flow_stn_id, tethys_obj, ds_ids_str, c_method):
    if isinstance(flow_stn_id, str):
        tethys = decode_obj(tethys_obj)
        ds_id = orjson.loads(ds_ids_str)['measured'][c_method]

        ts1 = get_results(tethys, ds_id, flow_stn_id)

        ts1_obj = encode_obj(ts1)

        return ts1_obj


@app.callback(
    Output('flow_nat', 'data'),
    [Input('sites', 'value')],
    [State('tethys', 'data'), State('ds_ids', 'data'), State('method_dd', 'value')])
# @cache.memoize()
def get_flow_nat_data(flow_stn_id, tethys_obj, ds_ids_str, c_method):
    if isinstance(flow_stn_id, str):
        tethys = decode_obj(tethys_obj)
        ds_id = orjson.loads(ds_ids_str)['naturalised'][c_method]

        ts1 = get_results(tethys, ds_id, flow_stn_id)

        ts1_obj = encode_obj(ts1)

        return ts1_obj


@app.callback(
    Output('allocation', 'data'),
    [Input('sites', 'value')],
    [State('tethys', 'data'), State('ds_ids', 'data'), State('last_month', 'data'), State('last_year', 'data')])
# @cache.memoize()
def get_allocation_data(flow_stn_id, tethys_obj, ds_ids_str, last_month, last_year):
    if isinstance(flow_stn_id, str):
        tethys = decode_obj(tethys_obj)
        ds_id = orjson.loads(ds_ids_str)['allocation']

        ts1 = get_results(tethys, ds_id, flow_stn_id, last_year, last_month)

        ts1_obj = encode_obj(ts1)

        return ts1_obj


@app.callback(
    Output('abstraction', 'data'),
    [Input('sites', 'value')],
    [State('tethys', 'data'), State('ds_ids', 'data'), State('last_month', 'data'), State('last_year', 'data')])
# @cache.memoize()
def get_abstraction_data(flow_stn_id, tethys_obj, ds_ids_str, last_month, last_year):
    if isinstance(flow_stn_id, str):
        tethys = decode_obj(tethys_obj)
        ds_id = orjson.loads(ds_ids_str)['abstraction']

        ts1 = get_results(tethys, ds_id, flow_stn_id, last_year, last_month)

        ts1_obj = encode_obj(ts1)

        return ts1_obj


@app.callback(Output('plots', 'children'),
              [Input('plot_tabs', 'value'), Input('flow_meas', 'data'), Input('flow_nat', 'data'), Input('allocation', 'data'), Input('abstraction', 'data')],
              [State('last_month', 'data'), State('last_year', 'data'), State('sites', 'value')])
@cache.memoize()
def render_plot(tab, flow_meas_str, flow_nat_str, allo_str, use_str, last_month, last_year, stn_id):

    # print(flow_stn_id)

    info_str = """
            ### Intro
            This is the [Environment Southland](https://www.es.govt.nz/) streamflow naturalisation, surface water usage, and surface water allocation dashboard.

            ### Brief Guide
            #### Selecting datasets
            The datasets are broken into two groups: **Recorder** and **Gauging** data. Recorder data have been used directly, while gauging data have been correlated to recorder sites to simulate  recorder data. There is also an option to select only the active flow sites (those with data in the last year) and all flow sites.

            #### Map
            The map shows the available streamflow sites given the prior selection on the left. **Click on a site** and the map will show the upstream catchment and the associated water abstraction sites (WAPs) in black.

            #### Data tabs
            The other tabs have plots for various use cases.

            ##### Flow duration
            The **Flow duration** plot orders the entire record from highest to lowest to indicate how often a particular flow is exceeded. The measured and naturalised flows are plotted together for comparisons, although in many cases they are very similar.

            ##### Cumulative flow
            The **Cumulative flow** plot accumulates the flow for each year in the record to show how this year compares to previous years.

            ##### Hydrograph
            The **Hydrograph** plot shows the entire record of a particular site.

            ##### Allocation
            The **Allocation** plot shows the current surface water allocation estimated by the upstream consents, the associated water usage, and the flow at 30% of the Q95 for that site. In many cases, the flow at 30% of the Q95 is the surface water allocation limit for rivers.

            ### Gauging correlations
            Naturalised streamflows have been estimated at all surface water recorder sites and gaugings sites with at least 12 gaugings. The gauging data has been automatically correlated to nearby recorder sites to generate continuous time series datasets. The correlation parameters and accuracies are shown below the site summaries below the plots. These include the normalised root mean square error (NRMSE), mean absolute error (MANE), adjusted R^2 (Adj R2), number of observations used in the correlation, the recorder sites used in the correlation, and the F value that was used to determine the appropriate recorder sites for the correlation.

            ### More info
            A more thorough description of the streamflow naturalisation method can be found [here](https://github.com/mullenkamp/nz-flow-naturalisation/blob/main/README.rst).
        """
    print(tab)

    if tab == 'info_tab':
        fig1 = info_str

    else:

        if (flow_meas_str is not None) and (flow_nat_str is not None) and (allo_str is not None) and (use_str is not None):
            if len(flow_meas_str) > 1:

                if tab == 'fd_plot':
                    flow_meas = decode_obj(flow_meas_str)
                    flow_nat = decode_obj(flow_nat_str)

                    fig1 = fig_flow_duration(flow_meas, flow_nat)

                elif tab == 'cf_plot':
                    flow_meas = decode_obj(flow_meas_str)

                    fig1 = fig_cumulative_flow(flow_meas)

                elif tab == 'hydro_plot':
                    flow_meas = decode_obj(flow_meas_str)
                    flow_nat = decode_obj(flow_nat_str)

                    fig1 = fig_hydrograph(flow_meas, flow_nat)

                elif tab == 'allo_plot':
                    flow_meas = decode_obj(flow_meas_str)
                    allo = decode_obj(allo_str)
                    use = decode_obj(use_str)

                    fig1 = fig_allo_use(allo, use, flow_meas, stn_id)
            else:
                fig1 = info_str
        else:
            fig1 = info_str

    if isinstance(fig1, str):
        return dcc.Markdown(fig1)
    else:
        fig = dcc.Graph(
                # id = 'plots',
                figure = fig1,
                config={"displaylogo": False, 'scrollZoom': True, 'showLink': False}
                )

        return fig


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=80)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8080)
