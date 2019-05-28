import lifelines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def timegap(x, y, timedelta='D'):
    """
    Calculate the number of ``timedelta``'s between date x and y. Order of
    the dates is unimportant.

    Parameters
    ----------
    x : str, date-like, or Series-like
        First date, or series of dates.
    y : str, date-like, or Series-like
        Second date, or series of dates.
    timedelta : str
        Parameter to pass to :meth:`np.timedelta64` for calculating the number
        of steps. Defaults to 'D' for 'days'.

    Returns
    -------
    gap : float or 1D array of floats
        Gap(s) between dates.

    See Also
    --------
    :meth:`daygap`
    :meth:`weekgap`
    :meth:`monthgap`
    :meth:`yeargap`
    """

    x, y = pd.to_datetime(x), pd.to_datetime(y)

    return np.abs((x - y) / np.timedelta64(1, timedelta))


def daygap(x, y):
    """
    Calculate the number of days between date x and y. Order of
    the dates is unimportant.

    Parameters
    ----------
    x : str, date-like, or Series-like
        First date, or series of dates.
    y : str, date-like, or Series-like
        Second date, or series of dates.

    Returns
    -------
    gap : float or 1D array of floats
        Gap(s) between dates.

    See Also
    --------
    :meth:`timegap`
    :meth:`weekgap`
    :meth:`monthgap`
    :meth:`yeargap`
    """
    return timegap(x, y)


def weekgap(x, y):
    """
    Calculate the number of weeks between date x and y. Order of
    the dates is unimportant.

    Parameters
    ----------
    x : str, date-like, or Series-like
        First date, or series of dates.
    y : str, date-like, or Series-like
        Second date, or series of dates.

    Returns
    -------
    gap : float or 1D array of floats
        Gap(s) between dates.

    See Also
    --------
    :meth:`timegap`
    :meth:`daygap`
    :meth:`monthgap`
    :meth:`yeargap`
    """
    return timegap(x, y, timedelta='W')


def monthgap(x, y):
    """
    Calculate the number of months between date x and y. Order of
    the dates is unimportant.

    Parameters
    ----------
    x : str, date-like, or Series-like
        First date, or series of dates.
    y : str, date-like, or Series-like
        Second date, or series of dates.

    Returns
    -------
    gap : float or 1D array of floats
        Gap(s) between dates.

    See Also
    --------
    :meth:`timegap`
    :meth:`daygap`
    :meth:`weekgap`
    :meth:`yeargap`
    """
    return timegap(x, y, timedelta='M')


def yeargap(x, y):
    """
    Calculate the number of years between date x and y. Order of
    the dates is unimportant.

    Parameters
    ----------
    x : str, date-like, or Series-like
        First date, or series of dates.
    y : str, date-like, or Series-like
        Second date, or series of dates.

    Returns
    -------
    gap : float or 1D array of floats
        Gap(s) between dates.

    See Also
    --------
    :meth:`timegap`
    :meth:`daygap`
    :meth:`monthgap`
    :meth:`monthgap`
    """
    return timegap(x, y, timedelta='Y')


def recall_at_k(y_true, y_pred, k=10):
    """
    Calculate recall at k.

    Parameters
    ----------
    y_true : array-like
        binary 1D numpy array. 0 - patient did not come in, 1 - patient did
        come in.
    y_pred : array-like
        1D array of probability scores
    k : int
        recall point

    Returns
    -------
    recall at k : int
    """

    y_true = y_true[np.argsort(-y_pred)]
    pos_labels_at_k = np.sum(y_true[0:k] > 0)
    pos_labels = np.sum(y_true > 0)
    r_at_k = pos_labels_at_k / pos_labels

    return r_at_k


def kaplan_plot(dataframe, group_col=None, event_col='TTE',
                observed_col='OBS', xlim=None, ax=None):
    """
    Creates a Kaplan-Meier plot for each group in `group_col`

    Parameters
    ----------
    dataframe : DataFrame
        Data to use for plots
    group_col : str, optional
        Groups to plot. If not separating by group, use a column
        with a single string value
    event_col : str, optional
        Name of the time to event column
    observed_col : str, optional
        Name of the event censoring column. 1 = event observed, 0 otherwise
    xlim : int
        Length of x-axis for plot
    ax : axis, optional
        If adding to an existing plot, set this to the existing ax value

    Returns
    -------
    None
        Call to plt.plot() of Kaplan-Meier estimated survival curve
    """

    kmf = lifelines.KaplanMeierFitter()

    if group_col is not None:
        add = ' by ' + group_col
        for group in dataframe[group_col].unique():
            grp = (dataframe[group_col] == group)
            kmf.fit(dataframe.loc[grp, event_col],
                    event_observed=dataframe.loc[grp, observed_col],
                    label=group)
            if ax is None:
                ax = kmf.plot()
            else:
                ax = kmf.plot(ax=ax)
    else:
        add = ''
        kmf.fit(dataframe[event_col], event_observed=dataframe[observed_col])
        ax = kmf.plot()

    if xlim is not None:
        ax.set_xlim(left=0, right=xlim)

    plt.title('Estimated Survival Curve' + add)


def nelson_plot(dataframe, group_col=None, event_col='TTE', observed_col='OBS',
                bandwidth=8, xlim=None, ax=None):
    """
    Creates a Nelson-Aalen plot for each group in `group_col`

    Parameters
    ----------
    dataframe : DataFrame
        Data to use for plots
    group_col : str, optional
        If provided, groups data by this column before fitting to plot
    event_col : str, optional
        Name of the time to event column
    observed_col : str, optional
        Name of the event observed column. 1 - observed, 0 otherwise.
    bandwidth : int
        Bandwidth to use for Hazard estimate
    xlim : int
        Length of x-axis for plot
    ax : axis, optional
        If adding to an existing plot, set this to the existing ax value

    Returns
    -------
    None
        Call to plt.plot() of Nelson-Aalen Hazard estimate
    """

    naf = lifelines.NelsonAalenFitter()

    if group_col is not None:
        title_add = ' by ' + group_col
        for group in dataframe[group_col].unique():
            grp = (dataframe[group_col] == group)
            naf.fit(dataframe[event_col][grp],
                    event_observed=dataframe[observed_col][grp],
                    label=group)
            if ax is None:
                ax = naf.plot_hazard(bandwidth=bandwidth)
            else:
                ax = naf.plot_hazard(ax=ax, bandwidth=bandwidth)
    else:
        title_add = ''
        naf.fit(dataframe[event_col],
                event_observed=dataframe[observed_col],
                label='Overall Survival Trend')
        if ax is None:
            ax = naf.plot_hazard(bandwidth=bandwidth)
        else:
            ax = naf.plot_hazard(ax=ax, bandwidth=bandwidth)

    if xlim is not None:
        ax.set_xlim(left=0, right=xlim)

    plt.title('Estimated Hazard Rate' + title_add)


def calculate_survival_params(data, timedelta, id_column='CID',
                              time_column='time'):
    """
    Calculate the TTE column and OBS columns for survival analysis from
    transaction history data.

    Parameters
    ----------
    data : DataFrame
        Should be in the format of treatment_details.
    timedelta : str
        What to use for calculating ``TTE``; passed to :meth:`np.timdelta64`
    id_column: str
        Identifier column name.
    time_column: str
        Time column name.

    Returns
    -------
    DataFrame
        data with TTE and OBS columns
    """

    df = data.copy()
    df = df.sort_values([id_column, time_column])
    df['Next'] = df.groupby([id_column])[time_column].shift(-1)
    df['OBS'] = pd.notna(df['Next']) * 1
    end_date = np.max(df[time_column])
    df['Next'] = df['Next'].fillna(end_date)
    df['TTE'] = timegap(df['Next'], df[time_column], timedelta=timedelta)

    return df


def to_survival_frame(cox_model, x, id_column=None, keep_only_ints=True):
    """
    Build frame required for scoring customers with the Cox
    Proportional-Hazards model.

    Parameters
    ----------
    cox_model
        An object resulting from lifelines.CoxPHFitter().fit()
    x : DataFrame
        Data to build S(t) from. The index of this DataFrame should be
        the patient id's. Columns should be covariates for the cox model
    id_column : str, optional
        If provided, sets the index of ``x`` to this column.
    keep_only_ints : bool
        If True, subsets the event_at column to only integer entries.

    Returns
    -------
    DataFrame
        A DataFrame with a column `event_at`, which is the time aspect of
        the survival analysis, and a column for each ConsumerID in the index of
        the input data `x`
    """

    if id_column is not None:
        if x.shape[0] != x[id_column].nunique():
            raise ValueError('ID column contains duplicate entries.')
        x = x.set_index(id_column)

    surv_df = cox_model.predict_survival_function(x)\
                       .reset_index()
    colnames = list(surv_df.columns)
    colnames[0] = 'event_at'
    surv_df.columns = colnames

    if keep_only_ints:
        surv_df = surv_df[surv_df.event_at % 1 == 0]

    return surv_df


def quick_score(cox_model, x, w, id_column=None, lapse_col='LapseTime'):
    """
    Quickly score patients using a cox proportional-hazards model.

    Parameters
    ----------
    cox_model
        An object resulting from :meth:`lifelines.CoxPHFitter().fit()`
    x : DataFrame
       Data to score; should contain ID as index and Lapse Time as a column
    w : int
        Length of time to score
    id_column : str, optional
        Name of consumer ID column, if not passed as index
    lapse_col : str, optional
        Name of the Lapse Time column

    Returns
    -------
    DataFrame
        ConsumerID with a ``CoxScore`` column.
    """

    surv_df = to_survival_frame(cox_model, x, id_column).set_index('event_at')\
                                                        .transpose()

    # Instantiate NumPy indexer arrays
    start_selector = np.zeros((surv_df.shape[0], surv_df.shape[1]))
    end_selector = start_selector.copy()

    # Create indexers
    starts = np.array(x[lapse_col], dtype=int)
    start_indices = np.minimum(starts, start_selector.shape[1] - 1)
    end_indices = np.minimum(start_indices + w, end_selector.shape[1] - 1)

    # Create selection arrays using the indexers
    start_selector[np.arange(len(start_indices)), start_indices] = 1
    end_selector[np.arange(len(end_indices)), end_indices] = 1

    # Subset surv_df using the arrays and calculate probabilities
    start_probs = np.sum(start_selector * surv_df, axis=1)
    end_probs = np.sum(end_selector * surv_df, axis=1)
    scores = (start_probs - end_probs) / start_probs
    score_df = pd.DataFrame({'CID': x.index, 'CoxScore': scores})\
                 .reset_index(drop=True)\
                 .fillna(0)

    return score_df


def _roc_parts(actual, score):
    """Get all the bits needed for AUC analysis."""

    fpr, tpr, threshold = roc_curve(actual, score)
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr


def measure_roc_auc(actual, score):
    """
    Measure AUC for indicators according to scores

    Parameters
    ----------
    actual : Series or list-like
        Indicators for observed events.
    score : Series or list-like
        Scores given by model.

    Returns
    -------
    roc_auc : float
        AUC for receiver operating characteristic
    """

    return _roc_parts(actual, score)[0]


def roc_plot(indicators, predicted_score):
    """
    Plot the ROC curve along with AUC measurement.

    Parameters
    ----------
    indicators : str, optional
        Name of column for purchase indication. 0-consumer did not purchase,
        1-consumer did purchase
    predicted_score : str, optional
        Score derived from CPH model

    Returns
    -------
    None
        Call to plt.plot() of ROC
    """

    plt.title('Receiver Operating Characteristic')

    roc_auc, fpr, tpr = _roc_parts(indicators, predicted_score)

    plt.plot(fpr, tpr, 'b', label='AUC %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')

    plt.show()
