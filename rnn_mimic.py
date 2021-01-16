import os
import pickle
from time import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.data_handler import return_data
from modules.model import ICU_LSTM


def get_targets():
    return ['MI', 'SEPSIS', 'VANCOMYCIN']


def get_percentages():
    return [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]


def get_pickle_path():
    '''

    Returns: the path to the pickle folder with the filename structure
             {variable_name}_{target}.txt in this case

    '''
    return './pickled_objects/{0}_{1}.txt'


def return_loaded_model(model_name="kaji_mach_0"):
    return torch.load("./saved_models/{0}.h5".format(model_name))


def dump_pickle(variable, path):
    with open(path, 'wb') as file:
        pickle.dump(variable, file)


def load_pickle(path):
    with open(path, 'rb') as file:
        variable = pickle.load(file)
    return variable


def print_reports(Y_PRED, Y_VAL):
    print('Confusion Matrix Validation')
    print(confusion_matrix(Y_VAL, np.around(Y_PRED)))
    print('Validation Accuracy')
    print(accuracy_score(Y_VAL, np.around(Y_PRED)))
    print('ROC AUC SCORE VAL')
    print(roc_auc_score(Y_VAL, Y_PRED))
    print('CLASSIFICATION REPORT VAL')
    print(classification_report(Y_VAL, np.around(Y_PRED)))


def train(model_name="kaji_mach_0", target='MI', predict=False, return_model=False,
          n_percentage=1.0, time_steps=14, epochs=10, batch_size=16, lr=0.001, device='cpu'):
    """

  Training the model using parameter inputs

  Args:
  ----
  model_name : Parameter used for naming the checkpoint_dir

  Return:
  -------
  Nonetype. Fits model only. 

  """
    path = get_pickle_path()
    checkpoint_dir = "./saved_models/{0}".format(model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    X_TRAIN = load_pickle(path.format('X_TRAIN', target))
    Y_TRAIN = load_pickle(path.format('Y_TRAIN', target))
    X_VAL = load_pickle(path.format('X_VAL', target))
    Y_VAL = load_pickle(path.format('Y_VAL', target))
    no_feature_cols = load_pickle(path.format('no_feature_cols', target))

    # N_Samples x Seq_Length x N_Features
    X_TRAIN = X_TRAIN[0:int(n_percentage * X_TRAIN.shape[0])]  # Subsample if necessary
    Y_TRAIN = Y_TRAIN[0:int(n_percentage * Y_TRAIN.shape[0])]

    # mask = X_TRAIN == 0
    # seq_lengths = torch.tensor(np.where(mask.any(1), mask.argmax(1), X_TRAIN.shape[1]))

    x_train_tensor = torch.tensor(X_TRAIN, dtype=torch.float)
    y_train_tensor = torch.tensor(Y_TRAIN, dtype=torch.float)
    x_val_tensor = torch.tensor(X_VAL, dtype=torch.float)
    y_val_tensor = torch.tensor(Y_VAL, dtype=torch.float)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = ICU_LSTM(no_feature_cols, time_steps).to(device)
    optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.9)
    loss_fn = nn.BCELoss()
    writer = SummaryWriter(log_dir='./logs/{0}_{1}.log'.format(model_name, time()))

    best_val_loss = 1e10
    for epoch in range(1, epochs + 1):
        print(f'\rEpoch {epoch:02d} out of {epochs}', end=" ")

        model.train()
        train_loss = []
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, _ = model(x, None)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        writer.add_scalar('Loss/train', np.mean(train_loss), epoch)

        model.eval()
        val_loss = []
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                output, _ = model(x, None)
                val_loss.append(loss_fn(output, y).item())
        val_loss = np.mean(val_loss)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, checkpoint_dir + f'/{model_name}.h5')

    torch.save(model, './saved_models/{0}.h5'.format(model_name))

    if predict:
        print('\nPrinting reports for TARGET: {0}\n'.format(target))
        Y_BOOLMAT_VAL = load_pickle(path.format('y_boolmat_val', target))
        model = model.to('cpu')
        Y_PRED = model(x_val_tensor, None).detach().numpy()
        print_reports(Y_PRED[~Y_BOOLMAT_VAL], Y_VAL[~Y_BOOLMAT_VAL])

    if return_model:
        return model


def pickle_objects(target='MI', time_steps=14):
    (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
     X_TEST, Y_TEST, x_boolmat_test, y_boolmat_test,
     x_boolmat_val, y_boolmat_val) = return_data(balancer=True, target=target, pad=True,
                                                 split=True, time_steps=time_steps)

    features = return_data(return_cols=True, target=target, pad=True, split=True, time_steps=time_steps)

    file_path = get_pickle_path()
    dump_pickle(X_TRAIN, file_path.format('X_TRAIN', target))
    dump_pickle(X_VAL, file_path.format('X_VAL', target))
    dump_pickle(Y_TRAIN, file_path.format('Y_TRAIN', target))
    dump_pickle(Y_VAL, file_path.format('Y_VAL', target))
    dump_pickle(X_TEST, file_path.format('X_TEST', target))
    dump_pickle(Y_TEST, file_path.format('Y_TEST', target))
    dump_pickle(x_boolmat_test, file_path.format('x_boolmat_test', target))
    dump_pickle(y_boolmat_test, file_path.format('y_boolmat_test', target))
    dump_pickle(x_boolmat_val, file_path.format('x_boolmat_val', target))
    dump_pickle(y_boolmat_val, file_path.format('y_boolmat_val', target))
    dump_pickle(no_feature_cols, file_path.format('no_feature_cols', target))
    dump_pickle(features, file_path.format('features', target))


def main(create_data=False):
    """

    Args:
        create_data: If the data should be create, False by default

    Returns:

    """
    percentages = get_percentages()
    epochs = 13
    time_steps = 14

    if create_data:
        print('Creating  Datasets')
        for target in get_targets():
            pickle_objects(target=target, time_steps=time_steps)
            print(f'Created Datasets for {target}')
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Training Models using {device}')
        for target in get_targets():
            print(f'\nTraining {target}')
            for percentage in percentages:
                p = int(percentage * 100)
                model_name = f'kaji_mach_final_no_mask_{target}_pad14_{p}_percent'
                train(model_name=model_name, epochs=epochs, predict=False, device=device,
                      target=target, time_steps=time_steps, n_percentage=percentage)

                torch.cuda.empty_cache()
                print(f'\rFinished training on {percentage * 100}% of data')


if __name__ == "__main__":
    main()
