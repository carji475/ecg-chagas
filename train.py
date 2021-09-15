import json
import torch
import os
from tqdm import tqdm
from resnet import ResNet1d
from dataloader import ECGDatasetH5, ECGDataloaderH5
import torch.optim as optim
import numpy as np
import math
from sklearn.metrics import precision_recall_curve
from evaluate import compute_metrics


def train(ep, dataload):
    model.train()  # training mode (e.g. dropout enabled)
    total_loss = 0  # accumulated loss
    n_entries = 0  # accumulated number of data points

    # set up waitbar
    train_desc = "Epoch {epoch:2d}: train - Loss: {loss:.6f}"
    train_bar = tqdm(initial=0, leave=True,
                     total=math.ceil(len(dataload) / args.batch_size),
                     desc=train_desc.format(epoch=ep, loss=0),
                     position=0)  # progress bar

    # allocate space for storing the outputs
    train_outpts = np.zeros(
        dataload.dset.in_chagas[dataload.start_idx:dataload.end_idx].sum())
    end = 0

    # loop over batches
    for traces, diagnoses in dataload:
        traces, diagnoses = traces.to(device), diagnoses.to(device)  # use cuda if available

        # reset start index
        start = end

        # Reinitialize grad
        model.zero_grad()

        # Forward pass
        model_output = model(traces)  # predicted ages
        loss = loss_function(model_output, diagnoses)

        end = start + len(traces)
        train_outpts[start:end] = torch.nn.Sigmoid()(model_output)\
            .detach().cpu().numpy().flatten()

        # Backward pass
        loss.backward()

        # Optimize
        optimiser.step()

        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progess bar
        train_bar.desc = train_desc.format(epoch=ep, loss=total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries, train_outpts


def eval(ep, dataload):
    model.eval()  # evaluation mode (e.g. dropout disabled)
    total_loss = 0  # accumulated loss
    n_entries = 0  # accumulated number of data points

    # set up waitbar
    eval_desc = "Epoch {epoch:2d}: valid - Loss: {loss:.6f}"
    eval_bar = tqdm(initial=0, leave=True,
                    total=math.ceil(len(dataload) / args.batch_size),
                    desc=eval_desc.format(epoch=ep, loss=0), position=0)

    # allocate space for storing the outputs
    eval_outputs = np.zeros(
        dataload.dset.in_chagas[dataload.start_idx:dataload.end_idx].sum())
    end = 0

    # loop over the batches
    for traces, diagnoses in dataload:
        traces, diagnoses = traces.to(device), diagnoses.to(device)

        # reset start index
        start = end

        with torch.no_grad():  # disable gradient tracking
            # Forward pass
            model_output = model(traces)
            loss = loss_function(model_output, diagnoses)

            # store output
            end = start + len(traces)
            eval_outputs[start:end] = torch.nn.Sigmoid()(model_output)\
                .detach().cpu().numpy().flatten()

            # Update accumulated values
            total_loss += loss.detach().cpu().numpy()
            n_entries += len(traces)

            # Print result
            eval_bar.desc = eval_desc.format(epoch=ep, loss=total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries, eval_outputs


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""

    # Get precision-recall curve
    precision, recall, threshold = precision_recall_curve(y_true, y_score)

    # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
    f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))

    # Select threshold that maximize f1 score
    index = np.argmax(f1_score)
    opt_precision = precision[index]
    opt_recall = recall[index]
    opt_threshold = threshold[index - 1] if index != 0 else threshold[0] - 1e-10
    return opt_precision, opt_recall, opt_threshold


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict chagas from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 1)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                             'to fit into the given size. (default: 4096)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--valid_split', type=float, default=0.15,
                        help='fraction of the data used for validation (default: 0.15).')
    parser.add_argument('--data_tot', type=float, default=1.0,
                        help='fraction of the data used in total (default: 1.0).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+',
                        default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+',
                        default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--ptmdl', default=None,
                        help='pre-trained model path (default: None)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--examid_dset', default='exam_id',
                        help='exam id dataset in the hdf5 file.')
    parser.add_argument('--pos_weight', action='store_true',
                        help='use positive weighting in the loss. (default: False)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay in the optimisation. (default: 0)')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--path_to_data',
                        default='../data/code15/exams_part0.hdf5',
                        help='path to file containing ECG traces for training')
    parser.add_argument('--path_to_chagas', default='../data/chagas.csv',
                        help='path to csv file containing chagas diagnoses')

    args, unk = parser.parse_known_args()

    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    print(args)

    # Set device
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    folder = args.folder

    # Generate output folder if needed
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    # Save config file
    with open(os.path.join(args.folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    # =============== Build data loaders =======================================#
    tqdm.write("Building data loaders...")

    dset = ECGDatasetH5(
        path=args.path_to_data,
        traces_dset=args.traces_dset,
        exam_id_dset=args.examid_dset,
        ids_dset=None,
        path_to_chagas=args.path_to_chagas
    )
    train_end = round(args.data_tot * len(dset))
    n_valid = round(args.valid_split * train_end)
    valid_loader = ECGDataloaderH5(dset, args.batch_size, start_idx=0,
                                   end_idx=n_valid)
    train_loader = ECGDataloaderH5(dset, args.batch_size, start_idx=n_valid,
                                   end_idx=train_end)

    # true train & validation labels (to be used when computing metrics)
    train_true = train_loader.getfullbatch(attr_only=True)
    valid_true = valid_loader.getfullbatch(attr_only=True)

    # save some data info
    pptrain = train_loader.getfullbatch(attr_only=True).sum() / len(
        train_loader)
    ppvalid = valid_loader.getfullbatch(attr_only=True).sum() / len(
        valid_loader)
    data_info = 'Train positive proportion: {}' \
                '\nValid positive proportion: {}'.format(pptrain, ppvalid)
    file = open(os.path.join(args.folder, 'data_info.txt'), 'w+')
    file.write(data_info)
    file.close()

    # =============== Define model =============================================#
    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1  # two classes, but just need one output before sigmoid
    if args.ptmdl is None:
        model = ResNet1d(input_dim=(N_LEADS, args.seq_length),  # (12, 4096)
                         blocks_dim=list(
                             zip(args.net_filter_size, args.net_seq_lengh)),
                         n_classes=N_CLASSES,
                         kernel_size=args.kernel_size,
                         dropout_rate=args.dropout_rate)
    else:
        # Get checkpoint
        # model2 is saved in the old format so that it works with the older
        # pytorch version on hydra
        ckpt = torch.load(os.path.join(args.ptmdl, 'model2.pth'),
                          map_location=lambda storage, loc: storage)
        # Get config
        config = os.path.join(args.ptmdl, 'config.json')
        with open(config, 'r') as f:
            config_dict = json.load(f)
        model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                         blocks_dim=list(zip(config_dict['net_filter_size'],
                                             config_dict['net_seq_length'])),
                         n_classes=N_CLASSES,
                         kernel_size=config_dict['kernel_size'],
                         dropout_rate=args.dropout_rate)
        ckpt['model']['conv1.weight'] = torch.zeros([64, 12, 17])
        torch.nn.init.normal_(ckpt['model']["conv1.weight"])
        ckpt['model']["lin.weight"] = torch.zeros([1, 5120])
        torch.nn.init.normal_(ckpt['model']["lin.weight"])
        ckpt['model']["lin.bias"] = torch.zeros(1)
        model.load_state_dict(ckpt['model'])

    model.to(device=device)
    tqdm.write("Done!")

    # =============== Define loss function =====================================#
    pos_weight = None if not(args.pos_weight) else dset.get_weights()
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # =============== Define optimiser =========================================#
    tqdm.write("Define optimiser...")
    optimiser = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    tqdm.write("Done!")

    # =============== Define lr scheduler ======================================#
    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                     patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)
    tqdm.write("Done!")

    # =============== Train model ==============================================#
    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf

    # create data frame to store the results in
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'train_opt_thres', 'train_acc_bal', 'train_f1',
                                    'train_matthew', 'train_acc', 'train_prec',
                                    'train_recall',
                                    'valid_opt_thres', 'valid_acc_bal', 'valid_f1',
                                    'valid_matthew', 'valid_acc', 'valid_prec',
                                    'valid_recall'])

    # loop over epochs
    for ep in range(start_epoch, args.epochs):
        # compute losses and outputs for training & validation data
        train_loss, train_outputs = train(ep, train_loader)
        valid_loss, valid_outputs = eval(ep, valid_loader)

        # get optimal classification threshold
        _, _, train_opt_thres = get_optimal_precision_recall(train_true, train_outputs)
        _, _, valid_opt_thres = get_optimal_precision_recall(valid_true, valid_outputs)

        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),  # model state
                        'valid_loss': valid_loss,
                        'opt_threshold': valid_opt_thres,
                        'optimiser': optimiser.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss

        # Get learning rate
        learning_rate = optimiser.param_groups[0]["lr"]

        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break

        # Print message
        tqdm.write('Epoch {epoch:2d}: \tTrain Loss {train_loss:.6f} '
                   '\tValid Loss {valid_loss:.6f} \tLearning Rate {lr:.7f}\t'
                   .format(epoch=ep, train_loss=train_loss,
                           valid_loss=valid_loss, lr=learning_rate))

        # get metrics
        # train
        train_pred = (train_outputs - train_opt_thres + 0.5).round().astype(int)
        train_accuracy_balanced, train_f1_score, train_matthew, train_accuracy, \
            train_precision, train_recall = \
            compute_metrics(train_true.astype(int), train_pred)
        # valid
        valid_pred = (valid_outputs - valid_opt_thres + 0.5).round().astype(int)
        valid_accuracy_balanced, valid_f1_score, valid_matthew, valid_accuracy, \
            valid_precision, valid_recall = \
            compute_metrics(valid_true.astype(int), valid_pred)

        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                  "valid_loss": valid_loss,
                                  "lr": learning_rate,
                                  "train_opt_thres": train_opt_thres,
                                  "train_acc_bal": train_accuracy_balanced,
                                  "train_f1": train_f1_score,
                                  "train_matthew": train_matthew,
                                  "train_acc": train_accuracy,
                                  "train_prec": train_precision,
                                  "train_recall": train_recall,
                                  "valid_opt_thres": valid_opt_thres,
                                  "valid_acc_bal": valid_accuracy_balanced,
                                  "valid_f1": valid_f1_score,
                                  "valid_matthew": valid_matthew,
                                  "valid_acc": valid_accuracy,
                                  "valid_prec": valid_precision,
                                  "valid_recall": valid_recall},
                                 ignore_index=True)  # can only append a dict if ignore_index=True
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)

        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")
