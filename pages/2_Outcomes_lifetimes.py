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
import copy

import plotly.graph_objs as go
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots

import stroke_utilities.scenario

# Custom functions:
from utilities_msm.fixed_params import page_setup
# from utilities_msm.inputs import \
#     write_text_from_file
from utilities_msm.plot_maps import plot_scenario_outcomes
# Imports from the stroke_outcome package:
from stroke_outcome.discrete_outcome import Discrete_outcome
import stroke_outcome.outcome_utilities as outcome_utilities


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Title:
    st.markdown('# Outcomes + lifetime')

    # Import reference data
    mrs_dists, mrs_dists_notes = (
        outcome_utilities.import_mrs_dists_from_file())
    utility_dists, utility_dists_notes = (
        outcome_utilities.import_utility_dists_from_file())
    utility_weights = utility_dists.loc['Wang2020'].values

    mrs_colours = [
        "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
        "DarkSlateGray"  # mRS=6
        ]

    # Create patient data
    # Set random seed for repeatability:
    np.random.seed(42)

    # Time when treatment has no effect:
    ivt_time_no_effect_mins = 378.0
    mt_time_no_effect_mins = 480.0

    cols_treatment_time = st.columns([0.2, 0.8])
    with cols_treatment_time[0]:
        # All patients share these same treatment times:
        # time_to_ivt_mins = 90.0
        # time_to_mt_mins = 120.0
        time_to_ivt_mins = st.number_input(
            'Choose a treatment time in minutes',
            min_value=0,
            max_value=500,
            value=90,
            step=15
        )
    if time_to_ivt_mins > ivt_time_no_effect_mins:
        time_to_ivt_mins = ivt_time_no_effect_mins


    # Numbers of patients with each stroke type:
    n_nlvo = 65
    n_lvo = 35
    n_total = n_lvo + n_nlvo

    # # Store the patient details in this dictionary:
    # outcome_inputs_dict = dict(
    #     # Mix of LVO and nLVO:
    #     stroke_type_code=np.array([2]*n_lvo + [1]*n_nlvo),
    #     # Onset to needle time is fixed to 90mins:
    #     onset_to_needle_mins=np.full(n_total, time_to_ivt_mins),
    #     # Randomly pick whether IVT is chosen with around 15% yes:
    #     ivt_chosen_bool=np.random.binomial(1, 0.15, n_total) == 1,
    #     # Onset to puncture time is fixed to 120mins:
    #     onset_to_puncture_mins=np.full(n_total, time_to_mt_mins),
    #     # Nobody receives MT:
    #     mt_chosen_bool=np.full(n_total, 0),
    #     # Randomly pick whether MT is chosen for LVOs with around 30% yes:
    #     # mt_chosen_bool=np.concatenate(
    #     #     (np.random.binomial(1, 0.3, n_lvo), [0]*n_nlvo)) == 1,
    #     # Randomly pick some pre-stroke mRS scores from 0 to 5:
    #     mrs_pre_stroke=np.random.choice(np.arange(6), size=n_total)
    #     # ^ this should be using the pre-stroke mRS distribution from the referenec data - update me please
    # )

    # # Intitialise the outcome model:
    # discrete_outcome = Discrete_outcome(
    #     mrs_dists,
    #     n_total,
    #     utility_weights,
    #     # ivt_time_no_effect_mins=378.0,
    #     # mt_time_no_effect_mins=480.0
    # )

    # # Import patient array data into a dictionary called "trial".
    # for key in discrete_outcome.trial.keys():
    #     if key in outcome_inputs_dict.keys():
    #         discrete_outcome.trial[key].data = outcome_inputs_dict[key]

    # # Calculate outcomes:
    # outcomes_by_stroke_type, full_cohort_outcomes = (
    #     discrete_outcome.calculate_outcomes())

    # # Make a copy of the results:
    # outcomes_by_stroke_type = copy.copy(outcomes_by_stroke_type)
    # full_cohort_outcomes = copy.copy(full_cohort_outcomes)

    # st.write(full_cohort_outcomes)
    # each_patient_mrs_post_stroke = full_cohort_outcomes['each_patient_mrs_post_stroke']
    # each_patient_mrs_not_treated = full_cohort_outcomes['each_patient_mrs_not_treated']


    # Get lifetime model results from file.
    df_lifetime = pd.read_csv('./data_msm/lifetime_outcomes_results.csv')


    # Don't need the outcome model for this? Just the mRS distribution at some time?
    mrs_probs_cumulative, mrs_logodds = outcome_utilities.extract_mrs_probs_and_logodds(mrs_dists)

    treated_probs_cumulative, treated_odds, treated_logodds = outcome_utilities.calculate_mrs_dist_at_treatment_time(
        np.array(time_to_ivt_mins).reshape(1, 1),
        ivt_time_no_effect_mins,
        mrs_logodds['t0_treatment_lvo_ivt'],
        mrs_logodds['no_treatment_lvo'],
        final_value_is_mrs6=True
    )

    treated_probs_cumulative = treated_probs_cumulative.flatten()
    not_treated_probs_cumulative = mrs_probs_cumulative['no_treatment_lvo']

    # Turn from cumulative to non-cumulative probs:
    not_treated_probs = np.diff(not_treated_probs_cumulative, prepend=0.0)
    treated_probs = np.diff(treated_probs_cumulative, prepend=0.0)


    with cols_treatment_time[1]:
        # Show mRS dists at this treatment time:

        fig = go.Figure()
        # (not) treated cumulative probability:
        tcp = 0.0
        ntcp = 0.0
        for mrs in range(7):
            fig.add_trace(go.Bar(
                x=[not_treated_probs[mrs]],
                base=ntcp,
                offset=-0.4,
                width=0.8,
                y=['Not treated'],
                orientation='h',
                marker_color=mrs_colours[mrs],
                showlegend=False,
                name=f'mRS {mrs}',
                customdata=np.stack([[not_treated_probs[mrs]]], axis=-1),
                hovertemplate='%{customdata[0]:.3f}'
            ))
            fig.add_trace(go.Bar(
                x=[treated_probs[mrs]],
                base=tcp,
                offset=-0.4,
                width=0.8,
                y=['Treated'],
                orientation='h',
                marker_color=mrs_colours[mrs],
                name=f'mRS {mrs}',
                customdata=np.stack([[treated_probs[mrs]]], axis=-1),
                hovertemplate='%{customdata[0]:.3f}'
            ))
            tcp += treated_probs[mrs]
            ntcp += not_treated_probs[mrs]
        st.plotly_chart(fig)


    # Need to know which patients have which age and sex...
    # Import SSNAP statistics by age and sex:
    df_proportions = pd.read_csv('./data_msm/ssnap_age_sex_summary.csv')

    # Get colours from the plotly colour scale:
    age_colours = sample_colorscale(
        'viridis',
        np.linspace(0, 1, len(sorted(set(df_proportions['age']))))
        )


    st.markdown(
        '''
        In the SSNAP data, the proportions of patients in each
        age and sex category are shown in the following bar chart.
        In this app, these proportions are assumed to apply to
        each mRS score identically.
        '''
    )
    # Plot age and sex proportions:
    fig = go.Figure()
    cw = 0.0
    for a, age in enumerate(sorted(list(set(df_proportions['age'])))):
        colour = age_colours[a]
        for sex in ['Male', 'Female']:
            mask = (
                (df_proportions['age'] == age) &
                (df_proportions['sex_label'] == sex)
            )
            width = df_proportions['proportion'][mask].values[0]

            custom_data = np.stack([[sex], [width], [age]], axis=-1)
            ht = (''.join([
                '%{customdata[0]}<br>',
                'Age %{customdata[2]}<br>',
                'Proportion: %{customdata[1]:.3f}<br>',
                '<extra></extra>'
            ]))

            # y = 1 if sex == 'Male' else 0

            fig.add_trace(go.Bar(
                orientation='h',
                x=[width],
                y=[sex],
                width=1.0,
                base=cw,
                offset=-0.5,
                customdata=custom_data,
                marker_color=colour,
                name = f'Age {age}',
                showlegend=(True if sex == 'Male' else False),
                hovertemplate=ht,
                # text=sex[0],
            ))

            cw += width
    # fig.update_traces(textangle=0, textposition='auto', cliponaxis=False)
    fig.update_layout(xaxis_title='Probability')
    st.plotly_chart(fig)


    # RESULTS
    cols_results = st.columns([0.3, 0.7])

    # Pick out some features to plot on the y-axis:
    y_feature_options = [
        'death_in_year_1_prob',
        'survival_median_years',
        'life_expectancy',
        'year_when_zero_survival',
        'qalys_total',
        'ae_count',
        'ae_discounted_cost',
        'nel_count',
        'nel_discounted_cost',
        'el_count',
        'el_discounted_cost',
        'care_years',
        'care_years_discounted_cost',
        'total_discounted_cost',
        'net_benefit',
    ]
    with cols_results[0]:
        # Let the user select which one to show:
        y_feature = st.selectbox(
            'Feature to plot',
            options=y_feature_options
        )

    #
    df_results = pd.DataFrame(columns=[
        'age',
        'sex',
        # 'proportion_this_age_and_sex',
        'mrs',
        'outcome',
        'treated_proportion',
        'not_treated_proportion'
        ])
    #
    c = 0  # Count
    for sex in ['Male', 'Female']:
        for age in sorted(list(set(df_proportions['age']))):
            prop_age_sex = df_proportions['proportion'][
                (df_proportions['age'] == age) &
                (df_proportions['sex_label'] == sex)
            ].values[0]
            for mrs in range(7):
                try:
                    lifetime_outcome = df_lifetime[y_feature][
                        (df_lifetime['age'] == age) &
                        (df_lifetime['sex_label'] == sex) &
                        (df_lifetime['mrs'] == mrs)
                    ].values[0]
                except IndexError:
                    # No patients survive with mRS=6 so it's not in
                    # the lifetime dataframe.
                    lifetime_outcome = 0.0

                prop_treated = (
                    prop_age_sex * treated_probs[mrs]
                )
                prop_not_treated = (
                    prop_age_sex * not_treated_probs[mrs]
                )

                df_results.loc[c] = [
                    age, sex, 
                    # prop_age_sex, 
                    mrs, lifetime_outcome, prop_treated, prop_not_treated
                ]

                # Iterate:
                c += 1

    # st.write(np.sum(treated_probs))
    # st.write(np.sum(mrs_probs['no_treatment_lvo']))

    # Mean results:
    df_results['treated_prop*outcome'] = (
        df_results['treated_proportion'] *
        df_results['outcome']
    )

    df_results['not_treated_prop*outcome'] = (
        df_results['not_treated_proportion'] *
        df_results['outcome']
    )
    df_results['diff'] = (df_results['treated_prop*outcome'] -
                          df_results['not_treated_prop*outcome'])

    df_results = df_results.sort_values(['mrs', 'age', 'sex'])

    # df_results['cumulative_treated_props'] = np.cumsum(np.append(0.0, df_results['treated_proportion']))[:-1]
    # df_results['cumulative_not_treated_props'] = np.cumsum(np.append(0.0, df_results['not_treated_proportion']))[:-1]

    with cols_results[1]:
        with st.expander('Results table'):
            st.dataframe(df_results)

    mean_outcome_treated = df_results['treated_prop*outcome'].sum()
    mean_outcome_not_treated = df_results['not_treated_prop*outcome'].sum()

    with cols_results[0]:
        st.markdown(f'Total without treatment: {mean_outcome_not_treated}')
        st.markdown(f'Total with treatment: {mean_outcome_treated}')
        st.markdown(f'Difference: {mean_outcome_treated - mean_outcome_not_treated}')

        st.write(df_results['outcome'].mean())

    st.markdown('## Difference by patient category')
    
    # fig = go.Figure()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.1, 0.8, 0.1])
    fig.update_layout(
        width=1200,
        height=1000,
        margin_l=0, margin_r=0, margin_b=0, margin_t=50
        )
    # (not) treated cumulative probability:
    tcp = 0.0
    ntcp = 0.0
    for mrs in range(7):
        fig.add_trace(go.Bar(
            x=[treated_probs[mrs]],
            base=tcp,
            offset=-0.5,
            width=1.0,
            y=['Treated'],
            orientation='h',
            marker_color=mrs_colours[mrs],
            name=f'mRS {mrs}',
            legendgroup='1'
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[not_treated_probs[mrs]],
            base=ntcp,
            offset=-0.5,
            width=1.0,
            y=['Not treated'],
            orientation='h',
            marker_color=mrs_colours[mrs],
            showlegend=False,
            name=f'mRS {mrs}'
        ), row=3, col=1)
        tcp += treated_probs[mrs]
        ntcp += not_treated_probs[mrs]

    add_age_to_legend = True
    all_bin_edges = []
    for mrs_t in range(7):
        for mrs_nt in range(7):

            bin_left_t = (
                treated_probs_cumulative[mrs_t - 1]
                if mrs_t > 0 else 0.0
            )
            bin_left_nt = (
                not_treated_probs_cumulative[mrs_nt - 1]
                if mrs_nt > 0 else 0.0
            )

            bin_size_t = treated_probs[mrs_t]
            bin_size_nt = not_treated_probs[mrs_nt]

            bin_right_t = bin_left_t + bin_size_t
            bin_right_nt = bin_left_nt + bin_size_nt

            all_bin_edges += [
                bin_left_t,
                bin_left_nt,
                bin_right_t,
                bin_right_nt
            ]

            if mrs_t != mrs_nt:
                # Find the overlap between these two mRS scores.
                if (bin_left_nt > bin_right_t) | (bin_left_t > bin_right_nt):
                    bin_overlap = 0.0
                else:
                    overlap_left = max([bin_left_nt, bin_left_t])
                    overlap_right = min([bin_right_nt, bin_right_t])
                    bin_overlap = overlap_right - overlap_left
                    # bin_overlap = (bin_right_t - bin_right_nt) - (bin_left_t - bin_left_nt)


                cw = overlap_left
                if bin_overlap != 0.0:
                    for a, age in enumerate(sorted(list(set(df_proportions['age'])))):
                        colour = age_colours[a]
                        for sex in ['Male', 'Female']:
                            mask = (
                                (df_proportions['age'] == age) &
                                (df_proportions['sex_label'] == sex)
                            )
                            prop_age_sex = df_proportions['proportion'][mask].values[0]
                            
                            width = prop_age_sex * bin_overlap

                            outcome_t = df_results['outcome'][
                                (df_results['age'] == age) &
                                (df_results['sex'] == sex) &
                                (df_results['mrs'] == mrs_t)
                                ].values[0]
                    
                            outcome_nt = df_results['outcome'][
                                (df_results['age'] == age) &
                                (df_results['sex'] == sex) &
                                (df_results['mrs'] == mrs_nt)
                                ].values[0]
                            
                            outcome_diff = (outcome_t - outcome_nt)
                            
                            custom_data = np.stack([[sex], [width], [age]], axis=-1)
                            ht = (''.join([
                                '%{customdata[0]}<br>',
                                'Age %{customdata[2]}<br>',
                                'Proportion: %{customdata[1]:.5f}<br>',
                                'Difference: %{y:5g}',
                                '<extra></extra>'
                            ]))

                            fig.add_trace(go.Bar(
                                x=[cw],
                                y=[outcome_diff],
                                width=width,
                                base=0.0,
                                offset=0.0,
                                customdata=custom_data,
                                marker_color=colour,
                                name = f'Age {age}',
                                showlegend=(add_age_to_legend if sex == 'Male' else False),
                                legendgroup='2',
                                hovertemplate=ht
                            ), row=2, col=1)

                            cw += width
                add_age_to_legend = False


    for bin_edge in list(set(all_bin_edges)):
        fig.add_vline(x=bin_edge, row=2, col=1, line_color='grey', opacity=0.2)

    # Set legend location:
    fig.update_layout(legend=dict(
        orientation='h', #'h',
        # yanchor='top',
        y=-0.2,
        # xanchor='right',
        # x=1.0
    ))
    # Figure format:
    fig.update_layout(
        # xaxis_title='Age (years)',
        yaxis2_title=f'Difference in {y_feature}',
        xaxis3_title='Proportion'
    )

    fig.update_layout(
        xaxis_range=[0.0, 1.0],
        yaxis_range=[0.0, 1.0],
        xaxis2_range=[0.0, 1.0],
        xaxis3_range=[0.0, 1.0],
        yaxis3_range=[0.0, 1.0]
        )
    st.plotly_chart(fig)

    # def scatter_results(df, y_feature_display_name, col):
    #     """
    #     Make a plotly scatter plot for age, sex, mrs, and a feature.

    #     Inputs:
    #     -------
    #     df                     - pd.DataFrame. Results of the lifetime
    #                             outcomes model.
    #     y_feature_display_name - str. Column in the dataframe to plot
    #                             on the y-axis.
    #     col                    - str. Column for either mRS scores
    #                             (separate mRS model) or outcome type
    #                             (dichotomous model).
    #     """

    #     fig = go.Figure()
    #     fig.update_layout(
    #         width=800,
    #         height=600,
    #         margin_l=0, margin_r=0, margin_b=0, margin_t=50
    #         )

    #     # Get colours from the plotly colour scale:
    #     colours = sample_colorscale(
    #         'viridis',
    #         np.linspace(0, 1, len(sorted(set(df[col]))))
    #         )
    #     # Iterate over outcome and sex:
    #     for a, val in enumerate(sorted(set(df[col]))):
    #         colour = colours[a]
    #         for s, sex_label in enumerate(sorted(set(df['sex']))):
    #             m = 'circle' if sex_label == 'Female' else 'square'
    #             # Reduce the dataframe to just these rows:
    #             df_here = df[
    #                 (df[col] == val) &
    #                 (df['sex'] == sex_label)
    #             ]
    #             # Name for this trace in the legend:
    #             name_str = (f'{col} {val}: {sex_label}' if col == 'mrs' else
    #                         f'{val}: {sex_label}')
    #             # Add these patients to the plot:
    #             fig.add_trace(go.Scatter(
    #                 x=df_here['age'],
    #                 y=df_here[y_feature_display_name],
    #                 mode='lines+markers',
    #                 marker_color=colour,
    #                 marker_symbol=m,
    #                 marker_line_color='black',
    #                 marker_line_width=1.0,
    #                 line_color=colour,
    #                 name=name_str
    #             ))

    #     # Figure format:
    #     fig.update_layout(
    #         xaxis_title='Age (years)',
    #         yaxis_title=y_feature_display_name
    #     )
    #     st.plotly_chart(fig)


    # scatter_results(df_results, 'treated_prop*outcome', 'mrs')



    # ----- The end! -----


if __name__ == '__main__':
    main()
