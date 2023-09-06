# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:41:22 2022

@author: Micah Gross
"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\user\.spyder-py3\Laser"
# streamlit run Laser_app.py
# create requirements.txt in project folder
# pipreqs C:\Users\User\.spyder-py3\streamlit_apps\laser

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import openpyxl
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from io import StringIO
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#%%
def exp_func(x, ymax, tau, base=0, delay=0):
    '''
    x = df_2['time']
    ymax = df_2['speed'].max()
    '''
    return base + ymax*(1-np.exp(-(x-delay)/tau))

# test execution
# x = np.linspace(0,3,9)
# st.write(x)
# y = np.array([0,1.2,2.2,3,3.6,4,4.2,4.3,4.3])
# st.write(y)
# delay_est = 0.5# estimation of delay for setting the bounds of curve_fit
# tau_est = x[
#     min(np.where(y>=0.63*max(y))[0])
#     ] - delay_est# estimation of tau (as time to 63% of max) for setting the bounds of curve_fit

# bounds=(
#     [0.9*max(y), 0, -0.1, 0],
#     [1.2*max(y), 2*tau_est, 0.1, 2*delay_est]
#     )

# popt, pcov = curve_fit(
#     exp_func,
#     x,
#     y,
#     bounds=bounds
#     )

# st.write(exp_func(x, *popt))
# st.write('ok 1')
# fig = plt.figure()
# plt.plot(x, y, color='blue', label='speed')
# st.write('ok 2')
# plt.plot(x, exp_func(x, *popt), color='red', label='speed (model)')
# st.write('ok 3')
# st.pyplot(fig)
# st.write('ok 4')
#%%
def output_to_excel(Results):
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        Results.to_excel(
            writer,
            sheet_name='summary',
            index=False
            )
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)

    download_filename = 'Results_Sprint_Laser.xlsx'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{download_filename}">Download summary parameters as Excel file</a>'

#%%
if __name__=='__main__':
    st.legacy_caching.clear_cache()
    st.set_page_config(layout="wide")
    st.write("""
    # Sprint Laser
    """)
    st.sidebar.header('Options')
    Options = {'save_variables': True if 'C:\\Users' in os.getcwd() else False}
    if Options['save_variables']:
        for (_, _, files) in os.walk(os.path.join(os.getcwd(),'saved_variables')):
            for f in files:
                os.remove(os.path.join(os.getcwd(),'saved_variables', f))
            del files

    smoothing_increment = st.sidebar.selectbox(
        'smoothing increment (s)',
        [0.1, 0.2, 0.25, 0.5],
        index=2
        )
    # cols[0].write(smoothing_increment)
    # full_dist = cols[1].number_input(
    full_dist = st.sidebar.number_input(
        'full sprint distance (m)',
        value=25,
        step=5
        )
    # cols[1].write(full_dist)
    # start_dist = cols[2].number_input(
    start_dist = st.sidebar.number_input(
        'starting point of model (m)',
        value=1,
        step=1
        )
    # cols[2].write(start_dist)
    # model_dist = cols[3].number_input(
    model_dist = st.sidebar.number_input(
        'end point of model (m)',
        value=int(full_dist),
        )
    # cols[3].write(model_dist)
    # par_source = cols[4].radio(
    par_source = st.sidebar.radio(
        'take parameters from',
        ['smoothed data', 'model'],
        index=0
        )
    # cols[4].write(par_source)
    plot_col_nr = st.sidebar.number_input(
        'plots per row',
        min_value=1,
        max_value=3,
        value=3,
        )

    plot_pars_names = [
        'speed (smoothed, full)',
        'speed (smoothed, sprint distance)',
        'speed (modeled, to vmax)',
        'acceleration (smoothed, full)',
        'acceleration (smoothed, sprint distance)',
        'acceleration (modeled, to vmax)',
        ]
    plot_pars_all = [
        'speed_smoothed_full',
        'speed_smoothed_sprint',
        'speed_model_sprint',
        'accel_smooth_full',
        'accel_smooth_sprint',
        'accel_model_sprint',
        ]
    plot_pars_default = [
        'speed_smoothed_sprint',
        'speed_model_sprint',
        'accel_model_sprint',
        ]# 'speed_smoothed_full'
    st.sidebar.header('select parameters to plot')
    plot_pars = {
        par: st.sidebar.checkbox(
            plot_pars_names[i],
            True if par in plot_pars_default else False
            ) for i,par in enumerate(plot_pars_all)
        }
    upload_files = st.file_uploader(
        'upload raw data als csv file',
        accept_multiple_files=True
        )

    if upload_files != []:
        # cols = st.columns(5)
        # smoothing_increment = cols[0].selectbox(

        if Options['save_variables']:
            with open(os.path.join(os.getcwd(), 'saved_variables','smoothing_increment.json'), 'w') as fp:
                json.dump(smoothing_increment, fp)
            with open(os.path.join(os.getcwd(), 'saved_variables','full_dist.json'), 'w') as fp:
                json.dump(full_dist, fp)
            with open(os.path.join(os.getcwd(), 'saved_variables','start_dist.json'), 'w') as fp:
                json.dump(start_dist, fp)
            with open(os.path.join(os.getcwd(), 'saved_variables','model_dist.json'), 'w') as fp:
                json.dump(model_dist, fp)
            with open(os.path.join(os.getcwd(), 'saved_variables','par_source.json'), 'w') as fp:
                json.dump(par_source, fp)

        vt_container = st.empty()
        plot_col_nr = int(plot_col_nr if len(upload_files)>=plot_col_nr else len(upload_files))
        vt_cols = vt_container.columns(plot_col_nr)
        slider_container = st.empty()
        Results = pd.DataFrame()
        for f_nr, f in enumerate(upload_files):
            if Options['save_variables']:
                with open(os.path.join(os.getcwd(),'saved_variables',(f.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
                    fp.write(f.getbuffer())
                with open(os.path.join(os.getcwd(), 'saved_variables','Options.json'), 'w') as fp:
                    json.dump(Options, fp)
                
            f.seek(0)    
            df = pd.read_csv(f, sep=';', names=['time', 'position'])#.dropna()
            if Options['save_variables']:
                df.to_json(os.path.join(os.getcwd(), 'saved_variables','df.json'))
            
            #%%
            s_incr = np.round(float(df['time'].diff().dropna().mode()),3)
            df_smooth = df.rolling(window=int(smoothing_increment/s_incr), center=True).mean()#.dropna()
            df_chop = df.iloc[::int(smoothing_increment/s_incr/2)].iloc[1::2]
            df_2 = pd.concat(
                (
                    df_chop['time'],
                    df_smooth.loc[df_chop.index, 'position']
                    ),
                axis=1
                )# plt.figure() df_2['position'].plot()
            df_2['speed'] = np.gradient(df_2['position']) / np.gradient(df_2['time'])
            df_2['accel'] = np.gradient(df_2['speed']) / np.gradient(df_2['time'])

            plt.close('vt')
            pv_fig = plt.figure('vt')
            try:
                plt.title(f.name.split('.')[0])
            except:
                plt.title(files[0].split('_')[1].split('\\')[-1])
            if plot_pars['speed_smoothed_full']:
                plt.plot(
                    np.array(df_2['time']),
                    np.array(df_2['speed']),
                    # label='speed (smoothed)',
                    LineStyle=':',
                    color='blue'
                    )
            if plot_pars['speed_smoothed_sprint']:
                plt.scatter(
                    np.array(df_2[((df_2['position']>=start_dist) & (df_2['position']<=full_dist))]['time']),
                    np.array(df_2[((df_2['position']>=start_dist) & (df_2['position']<=full_dist))]['speed']),
                    label='speed (smoothed)',
                    facecolors='none', edgecolors='blue'
                    )

            on = df_2[df_2['position']<=model_dist][df_2['position']<start_dist].index.max()
            off = df_2[df_2['position']<=model_dist]['speed'].idxmax()
            x = df_2.loc[on:off, 'time']
            y = df_2.loc[on:off, 'speed']
            z = df_2.loc[on:off, 'accel']
            
            if plot_pars['speed_smoothed_sprint']:
                plt.scatter(np.array(x), np.array(y), color='blue')#, label='data points for model')
            delay_est = df_2[df_2['speed']>0.5]['time'].min()# estimation of delay for setting the bounds of curve_fit
            tau_est = df_2[df_2['speed']<=0.63*y.max()]['time'].max() - delay_est# estimation of tau (as time to 63% of max) for setting the bounds of curve_fit
            bounds=(
                [0.9*y.max(), 0, -0.1, 0],
                [1.2*y.max(), 2*tau_est, 0.1, 2*delay_est]
                )
            try:
                popt, pcov = curve_fit(
                    exp_func,
                    x.loc[:x.idxmax()],
                    y[y>0].loc[:x.idxmax()].dropna().sort_values(),
                    bounds=bounds
                    )
            except:
                continue
            if plot_pars['speed_model_sprint']:
                plt.plot(
                    np.array(x),
                    np.array(exp_func(x, *popt)),
                    color='red',
                    label='speed (model)'
                    )
            model_t = pd.Series(
                np.arange(
                    np.floor(popt[3]*(10**int(np.ceil(np.log10(1/s_incr))))) / (10**int(np.ceil(np.log10(1/s_incr)))),
                    df_2.loc[
                        df_2[df_2['position']<=full_dist].index.max(),
                        'time'
                        ] + s_incr,
                    s_incr
                    ),
                name='t'
                )
                    
            model_v = exp_func(
                model_t,
                *popt
                ).rename('v')

            model_d = (model_v * s_incr).cumsum() + df_2[df_2['time']>=popt[3]]['position'].min()
            model_a = pd.Series(np.gradient(model_v) / s_incr, name='a')

            if plot_pars['speed_model_sprint']:
                plt.plot(
                    np.array(model_t),
                    np.array(model_v),
                    color='red',
                    LineStyle='--'
                    )
            if plot_pars['accel_smooth_full']:
                plt.plot(
                    np.array(df_2['time']),
                    np.array(df_2['accel']),
                    label='accel (smoothed)',
                    LineStyle=':',
                    color='orange'
                    )
            if plot_pars['accel_smooth_sprint']:
                plt.scatter(
                    np.array(df_2[((df_2['position']>=start_dist) & (df_2['position']<=full_dist))]['time']),
                    np.array(df_2[((df_2['position']>=start_dist) & (df_2['position']<=full_dist))]['accel']),
                    facecolors='none', edgecolors='orange'
                    )
                plt.scatter(
                    np.array(x),
                    np.array(z),
                    color='orange'
                    )#, label='data points for model')
            if plot_pars['accel_model_sprint']:
                plt.plot(
                    np.array(model_t[((model_t>=x.iloc[0]) & (model_t<=x.iloc[-1]))]),
                    np.array(model_a[((model_t>=x.iloc[0]) & (model_t<=x.iloc[-1]))]),
                    color='green',
                    label='accel (model)'
                    )
                plt.plot(
                    np.array(model_t),
                    np.array(model_a),
                    color='green',
                    LineStyle='--'
                    )

            for ls,x in enumerate(np.unique([start_dist, model_dist, full_dist])):
                plt.axvline(df_2.loc[
                    df_2[df_2['position']<=x].index.max(),
                    'time'
                    ],
                    color='gray',
                    LineStyle=['--', ':', '-'][ls],
                    label=str(int(x))+' m'
                    )
            plt.ylabel('speed (m/s)')
            plt.xlabel('time (s)')

            if f_nr%3 == 0:
                plt.legend(loc = 'upper left')

            plt.ylim((-2, 1.2*df_2[df_2['position']<=full_dist]['speed'].max()))
            try:
                vt_cols[f_nr%3].pyplot(pv_fig)
            except:
                pass
            
            #%%
            result = pd.DataFrame(
                {
                    'Dateiname': f.name.split('.')[0],
                    'Sprintdistanz': int(full_dist),
                    'vmax': df_2[df_2['position']<=full_dist]['speed'].max(),
                    'Distanz vmax': df_2[df_2['position']<=full_dist].loc[
                        df_2[df_2['position']<=full_dist]['speed'].idxmax(),
                        'position'
                        ],
                    },
                index=[0]
                )
            
            v_result_raw = pd.DataFrame(
                {'vmax ' + str(int(d-5)) + '-' + str(int(d)) + 'm':
                  df_2[((df_2['position']>=d-5) & (df_2['position']<=d))]['speed'].max() for d in list(range(5,26,5))},
                index=[0]
                )
            v_result_model = pd.DataFrame(
                {'vmax ' + str(int(d-5)) + '-' + str(int(d)) + 'm':
                  model_v[((model_d>=d-5) & (model_d<=d))].max() for d in list(range(5,26,5))},
                index=[0]
                )

            a_result_raw = pd.DataFrame(
                {'amax ' + str(int(d)) + '-' + str(int(d+5)) + 'm':
                  df_2[((df_2['position']>=d-5) & (df_2['position']<=d))]['accel'].max() for d in list(range(5,26,5))},
                index=[0]
                )

            a_result_model = pd.DataFrame(
                {'amax ' + str(int(d)) + '-' + str(int(d+5)) + 'm':
                  model_a[((model_d>=d-5) & (model_d<=d))].max() for d in list(range(5,26,5))},
                index=[0]
                )
            
            if par_source == 'smoothed data':
                result = pd.concat((result, v_result_raw, a_result_raw), axis=1)
            elif par_source == 'model':
                result = pd.concat((result, v_result_model, a_result_model), axis=1)
                
            # Results = Results.append(result).reset_index(drop=True).sort_values(by=['Dateiname'])
            Results = pd.concat([Results, result])
            #%%
        st.dataframe(Results.style.format(subset=[col for col in Results.columns if 'max' in col], formatter="{:.2f}"))
        st.markdown(
            output_to_excel(
                Results.round({
                    col: 2 if 'max' in col else 0 for col in Results.columns 
                    })
                ),
            unsafe_allow_html=True
            )


