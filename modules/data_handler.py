import numpy as np
import pandas as pd

from modules.pad_sequences import PadSequences

ROOT = "./mimic_{0}_database/mapped_elements/"
FILE = "CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_plus_notes.csv"

def wbc_crit(x):
    return (x > 12 or x < 4) and x != 0


def temp_crit(x):
    return (x > 100.4 or x < 96.8) and x != 0


def get_synth_sequence(n_timesteps=14):
    """

  Returns a single synthetic data sequence of dim (bs,ts,feats)

  Args:
  ----
    n_timesteps: int, number of timesteps to build model for

  Returns:
  -------
    X: npa, numpy array of features of shape (1,n_timesteps,2)
    y: npa, numpy array of labels of shape (1,n_timesteps,1)

  """

    X = np.array([[np.random.rand() for _ in range(n_timesteps)], [np.random.rand() for _ in range(n_timesteps)]])
    X = X.reshape(1, n_timesteps, 2)
    y = np.array([0 if x.sum() < 0.5 else 1 for x in X[0]])
    y = y.reshape(1, n_timesteps, 1)
    return X, y


def load_data(balancer=True, target='MI', return_cols=False, tt_split=0.7,
              val_percentage=0.8, dataframe=False, time_steps=14,
              split=True, pad=True, seed=42, mimic_version=3):
    """
      Returns synthetic or real data depending on parameter

      Args:
      -----
          balance : whether or not to balance positive and negative time windows
          target : desired target, supports MI, SEPSIS, VANCOMYCIN or a known lab, medication
          return_cols : return columns used for this RNN
          tt_split : fraction of dataset to use fro training, remaining is used for test
          mask : 24 hour mask, default is False
          dataframe : returns dataframe rather than numpy ndarray
          time_steps : 14 by default, required for padding
          split : creates test train splits
          pad : by default is True, will pad to the time_step value
      Returns:
      -------
          Training and validation splits as well as the number of columns for use in RNN

      """
    np.random.seed(seed)
    df = pd.read_csv(ROOT.format(mimic_version) + FILE)

    # Delete features that make the task trivial
    if target == 'MI':
        df[target] = ((df['troponin'] > 0.4) & (df['CKD'] == 0)).apply(lambda x: int(x))
        toss = ['ct_angio', 'troponin', 'troponin_std', 'troponin_min', 'troponin_max', 'Infection', 'CKD']
    elif target == 'SEPSIS':
        df['hr_sepsis'] = df['heart rate'].apply(lambda x: 1 if x > 90 else 0)
        df['respiratory rate_sepsis'] = df['respiratory rate'].apply(lambda x: 1 if x > 20 else 0)
        df['wbc_sepsis'] = df['WBCs'].apply(wbc_crit)
        df['temperature f_sepsis'] = df['temperature (F)'].apply(temp_crit)
        df['sepsis_points'] = (df['hr_sepsis'] + df['respiratory rate_sepsis']
                               + df['wbc_sepsis'] + df['temperature f_sepsis'])
        df[target] = ((df['sepsis_points'] >= 2) & (df['Infection'] == 1)).apply(lambda x: int(x))
        del df['hr_sepsis']
        del df['respiratory rate_sepsis']
        del df['wbc_sepsis']
        del df['temperature f_sepsis']
        del df['sepsis_points']
        del df['Infection']
        toss = ['ct_angio', 'Infection', 'CKD']
    elif target == 'VANCOMYCIN':
        df['VANCOMYCIN'] = df['vancomycin'].apply(lambda x: 1 if x > 0 else 0)
        del df['vancomycin']
        toss = ['ct_angio', 'Infection', 'CKD']

    df = df.select_dtypes(exclude=['object'])

    COLUMNS = list(df.columns)
    COLUMNS = [i for i in COLUMNS if i not in toss]
    COLUMNS.remove(target)

    if 'HADM_ID' in COLUMNS:
        COLUMNS.remove('HADM_ID')
    if 'SUBJECT_ID' in COLUMNS:
        COLUMNS.remove('SUBJECT_ID')
    if 'YOB' in COLUMNS:
        COLUMNS.remove('YOB')
    if 'ADMITYEAR' in COLUMNS:
        COLUMNS.remove('ADMITYEAR')

    if return_cols:
        return COLUMNS

    if pad:
        pad_value = 0
        df = PadSequences().pad(df, 1, time_steps, pad_value=pad_value)
        print('There are {0} rows in the df after padding'.format(len(df)))

    if dataframe:
        return (df[COLUMNS + [target, "HADM_ID"]])

    MATRIX = df[COLUMNS + [target]].values
    MATRIX = MATRIX.reshape(int(MATRIX.shape[0] / time_steps), time_steps, MATRIX.shape[1])

    ## note we are creating a second order bool matirx
    bool_matrix = (~MATRIX.any(axis=2))
    MATRIX[bool_matrix] = np.nan
    MATRIX = PadSequences().ZScoreNormalize(MATRIX)
    ## restore 3D shape to boolmatrix for consistency
    bool_matrix = np.isnan(MATRIX)
    MATRIX[bool_matrix] = pad_value

    permutation = np.random.permutation(MATRIX.shape[0])
    MATRIX = MATRIX[permutation]
    bool_matrix = bool_matrix[permutation]

    X_MATRIX = MATRIX[:, :, 0:-1]
    Y_MATRIX = MATRIX[:, :, -1]

    x_bool_matrix = bool_matrix[:, :, 0:-1]
    y_bool_matrix = bool_matrix[:, :, -1]

    X_TRAIN = X_MATRIX[0:int(tt_split * X_MATRIX.shape[0]), :, :]
    Y_TRAIN = Y_MATRIX[0:int(tt_split * Y_MATRIX.shape[0]), :]
    Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

    x_train_boolmat = x_bool_matrix[0:int(tt_split * x_bool_matrix.shape[0])]
    y_train_boolmat = y_bool_matrix[0:int(tt_split * y_bool_matrix.shape[0])]
    y_train_boolmat = y_train_boolmat.reshape(y_train_boolmat.shape[0], y_train_boolmat.shape[1], 1)

    X_VAL = X_MATRIX[int(tt_split * X_MATRIX.shape[0]):int(val_percentage * X_MATRIX.shape[0])]
    Y_VAL = Y_MATRIX[int(tt_split * Y_MATRIX.shape[0]):int(val_percentage * Y_MATRIX.shape[0])]
    Y_VAL = Y_VAL.reshape(Y_VAL.shape[0], Y_VAL.shape[1], 1)

    x_val_boolmat = x_bool_matrix[
                    int(tt_split * x_bool_matrix.shape[0]):int(val_percentage * x_bool_matrix.shape[0])]
    y_val_boolmat = y_bool_matrix[
                    int(tt_split * y_bool_matrix.shape[0]):int(val_percentage * y_bool_matrix.shape[0])]
    y_val_boolmat = y_val_boolmat.reshape(y_val_boolmat.shape[0], y_val_boolmat.shape[1], 1)

    X_TEST = X_MATRIX[int(val_percentage * X_MATRIX.shape[0])::]
    Y_TEST = Y_MATRIX[int(val_percentage * X_MATRIX.shape[0])::]
    Y_TEST = Y_TEST.reshape(Y_TEST.shape[0], Y_TEST.shape[1], 1)

    x_test_boolmat = x_bool_matrix[int(val_percentage * x_bool_matrix.shape[0])::]
    y_test_boolmat = y_bool_matrix[int(val_percentage * y_bool_matrix.shape[0])::]
    y_test_boolmat = y_test_boolmat.reshape(y_test_boolmat.shape[0], y_test_boolmat.shape[1], 1)

    X_TEST[x_test_boolmat] = pad_value
    Y_TEST[y_test_boolmat] = pad_value

    if balancer:
        TRAIN = np.concatenate([X_TRAIN, Y_TRAIN], axis=2)
        print(np.where((TRAIN[:, :, -1] == 1).any(axis=1))[0])
        pos_ind = np.unique(np.where((TRAIN[:, :, -1] == 1).any(axis=1))[0])
        print(pos_ind)
        np.random.shuffle(pos_ind)
        neg_ind = np.unique(np.where(~(TRAIN[:, :, -1] == 1).any(axis=1))[0])
        print(neg_ind)
        np.random.shuffle(neg_ind)
        length = min(pos_ind.shape[0], neg_ind.shape[0])
        total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
        np.random.shuffle(total_ind)
        if target == 'MI':
            ind = pos_ind
        else:
            ind = total_ind
        X_TRAIN = TRAIN[ind, :, 0:-1]
        Y_TRAIN = TRAIN[ind, :, -1]
        Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

        x_train_boolmat = x_train_boolmat[ind]
        y_train_boolmat = y_train_boolmat[ind]

    no_feature_cols = X_TRAIN.shape[2]

    if split:
        return (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
                X_TEST, Y_TEST, x_test_boolmat, y_test_boolmat,
                x_val_boolmat, y_val_boolmat, x_train_boolmat, y_train_boolmat)

    else:
        return (np.concatenate((X_TRAIN, X_VAL), axis=0),
                np.concatenate((Y_TRAIN, Y_VAL), axis=0), no_feature_cols)
