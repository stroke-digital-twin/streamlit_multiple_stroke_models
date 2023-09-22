"""
Streamlit app template.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

import json
import plotly.graph_objs as go

import stroke_utilities.scenario

# Custom functions:
from utilities_msm.fixed_params import page_setup
from utilities_msm.inputs import \
    write_text_from_file
# Containers:
import utilities_msm.container_inputs
import utilities_msm.container_results
import utilities_msm.container_details


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Title:
    st.markdown('# Multiple models')
    st.markdown(':warning: This data is rubbish')

    # ----- Select region -----
    # Choose the region to be plotted.
    # List of stroke hospital names and coordinates:
    df_hospitals_and_lsoas = pd.read_csv(
        './data_msm/hospitals_and_lsoas.csv',
        # index_col='Postcode'
        ).sort_values('Stroke Team')

    # Select a hospital from the list:
    stroke_team_name = st.selectbox(
        'Select a hospital',
        options=df_hospitals_and_lsoas['Stroke Team'].copy()
        )
    # Pull out the data for just that hospital:
    series_hospitals_and_lsoas = df_hospitals_and_lsoas[
        df_hospitals_and_lsoas['Stroke Team'] == stroke_team_name].squeeze()
    # Coordinates for this team:
    stroke_team_coords = [
        series_hospitals_and_lsoas['long'],
        series_hospitals_and_lsoas['lat']
    ]
    # Postcode of this team (more consistent than name):
    stroke_team_postcode = series_hospitals_and_lsoas['Postcode']

    # ----- Geography -----

    # Select geojson files for this hospital.
    # Everything in the IVT catchment area:
    geojson_file = (
        './data_msm/nearest_hospital_geojson/' +
        f'LSOA_{stroke_team_postcode}.geojson'
        )
    with open(geojson_file) as f:
        stroke_team_geojson = json.load(f)
    
    # Sort out any problem polygons with coordinates in wrong order:
    from geojson_rewind import rewind
    stroke_team_geojson = rewind(stroke_team_geojson, rfc7946=False)
    # Everything in the MT catchment area:

    # Import data for regions in the catchment areas.
    # + Population
    # + Travel time to nearest IVT centre
    # + Travel time to nearest MT centre
    # + Name of nearest IVT centre
    # + Name of nearest MT centre
    df_lsoas_and_nearest_hospitals = pd.read_csv(
        './data_msm/lsoa_nearest_hospitals.csv')
    # Pick out only LSOAs whose nearest IVT centre is the
    # stroke team selected earlier.
    df_lsoas_and_nearest_hospitals = df_lsoas_and_nearest_hospitals[
        df_lsoas_and_nearest_hospitals['Nearest hospital postcode'] ==
        stroke_team_postcode
    ]
    lsoa_travel_times = df_lsoas_and_nearest_hospitals['Time (mins) to nearest hospital']

    # Admission numbers:
    df_admissions = pd.read_csv(
        './data_msm/admissions_2017-2019.csv',
        # index_col='area'
        )


    # ----- Pathway model -----
    # Import hospital performance statistics.
    # These patient statistics are quietly in the hospital performance:
    # + proportion with nLVO
    # + proportion with LVO
    # + proportion with other stroke types
    # because of the separate "admissions" parameter for each group.
    df_performance_scenarios = pd.read_csv(
        './data_msm/all_performance_scenarios.csv',
        index_col=0)
    # Pick out just this stroke team:
    stroke_team_id = 1
    df_performance_scenarios = df_performance_scenarios[
        df_performance_scenarios['stroke_team'] == stroke_team_id
    ]
    st.write('Note - currently those scenario performances are using the anonymised team names so I\'ve just picked out the first one no matter what inputs are given to streamlit.')
    # st.write(df_performance_scenarios)


    np.random.seed(42)

    # How many trials to run:
    n_trials = st.number_input(
        'Number of trials',
        min_value=0,
        max_value=200,
        value=100,
        step=10
        )
    if n_trials == 0:
        st.write('Running one trial')
        n_trials = 1

    results_df, outcome_results_columns, trial_columns = stroke_utilities.scenario.set_up_results_dataframe()


    scenario_list = sorted(list(set(df_performance_scenarios['scenario'])))
    df_scenario_outcomes_by_lsoa = pd.DataFrame()
    df_scenario_outcomes_by_lsoa['LSOA'] = df_lsoas_and_nearest_hospitals['LSOA']
    df_scenario_outcomes_by_lsoa.set_index('LSOA', inplace=True)

    for scenario in scenario_list:
        # Get data for one hospital.
        # Squeeze to convert DataFrame to Series.
        lvo_data = df_performance_scenarios[(
            (df_performance_scenarios['stroke_team'] == stroke_team_id) & 
            (df_performance_scenarios['stroke_type'] == 'lvo') &
            (df_performance_scenarios['scenario'] == scenario)
            )].copy().squeeze()
        nlvo_data = df_performance_scenarios[(
            (df_performance_scenarios['stroke_team'] == stroke_team_id) & 
            (df_performance_scenarios['stroke_type'] == 'nlvo') &
            (df_performance_scenarios['scenario'] == scenario)
            )].copy().squeeze()
        other_data = df_performance_scenarios[(
            (df_performance_scenarios['stroke_team'] == stroke_team_id) & 
            (df_performance_scenarios['stroke_type'] == 'other') &
            (df_performance_scenarios['scenario'] == scenario)
            )].copy().squeeze()


        # Set up trial results dataframe
        trial_df = pd.DataFrame(index=df_scenario_outcomes_by_lsoa.index)#columns=trial_columns)
        # Set up the pathways with this data...
        pathway_object_dict = stroke_utilities.scenario.set_up_pathway_objects(stroke_team_id, lvo_data, nlvo_data, other_data)

        for trial in range(n_trials):

            # ... run the pathways...
            combo_trial_dict = stroke_utilities.scenario.run_trial_of_pathways(pathway_object_dict)
            # ... overwrite the results so that nobody has thrombectomy...
            combo_trial_dict['mt_chosen_bool'] = np.array([0] * len(combo_trial_dict['mt_chosen_bool'])) == 1
            # ... and run the clinical outcome model.
            results_by_stroke_type, patient_array_outcomes = (
                stroke_utilities.scenario.run_discrete_outcome_model(combo_trial_dict))

            number_of_patients = len(combo_trial_dict['stroke_type_code'])
            # Patients' mRS if not treated..
            # st.write(patient_array_outcomes.keys())
            # dict_keys(['each_patient_mrs_dist_post_stroke', 'each_patient_mrs_post_stroke', 'each_patient_mrs_not_treated', 'each_patient_mrs_shift', 'each_patient_utility_post_stroke', 'each_patient_utility_not_treated', 'each_patient_utility_shift', 'mean_mrs_post_stroke', 'mean_mrs_not_treated', 'mean_mrs_shift', 'mean_utility', 'mean_utility_not_treated', 'mean_utility_shift'])
            mrs_not_treated = patient_array_outcomes['each_patient_mrs_not_treated']
            # Patients' mRS in this trial...
            mrs_post_stroke = patient_array_outcomes['each_patient_mrs_post_stroke']

            # Randomly (for now) assign patients to LSOAs:
            # np.random.seed(42)
            patient_array_lsoas = np.random.choice(
                df_lsoas_and_nearest_hospitals['LSOA'],
                size=number_of_patients
            )

            # Calculate mean change in outcome by LSOA:
            outcomes_by_lsoa = []
            for lsoa in df_scenario_outcomes_by_lsoa.index:
                inds_this_lsoa = np.where(patient_array_lsoas == lsoa)[0]

                mean_shift_mrs = np.mean(patient_array_outcomes['each_patient_mrs_shift'][inds_this_lsoa])

                # # How many patients were good outcomes? mRS 0 or 1.
                # n_good_baseline_here = len(np.where(mrs_not_treated[inds_this_lsoa] <= 1)[0])
                # n_good_post_stroke_here = len(np.where(mrs_post_stroke[inds_this_lsoa] <= 1)[0])
                # # Additional good outcomes:
                # n_good_additional_here = n_good_post_stroke_here - n_good_baseline_here
                # outcomes_this_lsoa = n_good_additional_here
                outcomes_by_lsoa.append(mean_shift_mrs)

            trial_df[trial] = outcomes_by_lsoa

        df_scenario_outcomes_by_lsoa[scenario] = trial_df.mean(axis=1).copy()

    # fig = go.Figure()
    # df_scenario_outcomes_by_lsoa.set_index('LSOA', inplace=True)

    outcome_vmin = df_scenario_outcomes_by_lsoa.min().min()
    outcome_vmax = df_scenario_outcomes_by_lsoa.max().max()

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=3, subplot_titles=scenario_list,
        specs=[[{"type": "choropleth"}]*3]*3
    )

    fig.update_layout(
        width=800,
        height=800
        )

    row = 1
    col = 1
    for scenario in scenario_list:

        fig.add_trace(go.Choropleth(
            geojson=stroke_team_geojson,
            # locations=df_lsoas_and_nearest_hospitals['LSOA'],
            # z=df_lsoas_and_nearest_hospitals['Time (mins) to nearest hospital'],
            locations=df_scenario_outcomes_by_lsoa.index,
            z=df_scenario_outcomes_by_lsoa[scenario],
            featureidkey='properties.LSOA11NM',
            coloraxis="coloraxis",
            # colorscale='Inferno',
            # autocolorscale=False
        ), row=row, col=col)
        # Mark the hospital location:
        fig.add_trace(go.Scattergeo(
            lon=[stroke_team_coords[0]],
            lat=[stroke_team_coords[1]],
            text=[stroke_team_name],
            marker_color=['black'],
            showlegend=False
            ), row=row, col=col)


        # # Remove LSOA borders:
        # fig.update_traces(marker_line_width=0, selector=dict(type='choropleth'))

        fig.update_layout(
            coloraxis_colorscale='Electric',
            coloraxis_colorbar_title_text='mRS shift',
            coloraxis_cmin=outcome_vmin,
            coloraxis_cmax=outcome_vmax,
            # row=row, col=col
            )

        col += 1
        if col > 3:
            col = 1
            row += 1
    geo_dict = dict(
            scope='world',
            projection=go.layout.geo.Projection(type = 'airy'),
            fitbounds='locations',
            visible=False)#, domain_row=row, domain_column=col)
    
    fig.update_layout(
        geo1 = geo_dict,
        geo2 = geo_dict,
        geo3 = geo_dict,
        geo4 = geo_dict,
        geo5 = geo_dict,
        geo6 = geo_dict,
        geo7 = geo_dict,
        geo8 = geo_dict,
        # geo9 = geo_dict,
    ) 
    st.plotly_chart(fig)


        
    #     for trial in range(n_trials):
    #         # ... run the pathways...
    #         combo_trial_dict = stroke_utilities.scenario.run_trial_of_pathways(pathway_object_dict)
    #         # ... overwrite the results so that nobody has thrombectomy...
    #         combo_trial_dict['mt_chosen_bool'] = np.array([0] * len(combo_trial_dict['mt_chosen_bool'])) == 1
    #         # ... and run the clinical outcome model.
    #         results_by_stroke_type, patient_array_outcomes = (
    #             stroke_utilities.scenario.run_discrete_outcome_model(combo_trial_dict))

    #         number_of_patients = len(combo_trial_dict['stroke_type_code'])
    #         # Patients' mRS if not treated..
    #         mrs_not_treated = patient_array_outcomes['each_patient_mrs_not_treated']
    #         # Patients' mRS in this trial...
    #         mrs_post_stroke = patient_array_outcomes['each_patient_mrs_post_stroke']
    #         # How many patients were good outcomes? mRS 0 or 1.
    #         n_good_baseline = len(np.where(mrs_not_treated <= 1)[0])
    #         n_good_post_stroke = len(np.where(mrs_post_stroke <= 1)[0])
    #         # Additional good outcomes:
    #         n_good_additional = n_good_post_stroke - n_good_baseline
    #         # Convert to outcomes per 1000 patients:
    #         n_baseline_good_per_1000 = n_good_baseline * (1000.0 / number_of_patients)
    #         n_additional_good_per_1000 = n_good_additional * (1000.0 / number_of_patients)
    
    #         result = stroke_utilities.scenario.gather_results_from_trial(
    #             trial_columns, combo_trial_dict, results_by_stroke_type,
    #             n_baseline_good_per_1000, n_additional_good_per_1000
    #             )
    #         trial_df.loc[trial] = result
    
    #     summary_trial_results = stroke_utilities.scenario.gather_summary_results_across_all_trials(outcome_results_columns, trial_df)
    #     summary_trial_results += [f'{stroke_team_id}']
    #     summary_trial_results += [scenario.replace(' + ', '_')]
        
    #     # add scenario results to results dataframe
    #     results_df.loc[f'{stroke_team_id} / {scenario}'] = summary_trial_results

    # st.write(results_df)
    # ----- Outcome model -----
    # 

    # ----- The end! -----

if __name__ == '__main__':
    main()