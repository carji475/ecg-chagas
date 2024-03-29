import json
import torch
import os
from tqdm import tqdm
from resnet import ResNet1d
from dataloader import ECGDatasetH5, ECGDataloaderH5
import torch.optim as optim
import numpy as np
from compute_metrics import compute_metrics


def train(ep, dataload, dataload2):
    model.train()  # training mode (e.g. dropout enabled)
    total_loss = 0  # accumulated loss
    n_batches = 0  # accumulated number of batches

    # set up waitbar
    train_desc = "Epoch {epoch:2d}: train - Loss: {loss:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(epoch=ep, loss=0),
                     position=0)  # progress bar

    # loop over batches
    for traces, diagnoses in dataload:
        traces2, diagnoses2 = next(dataload2)
        traces = torch.cat((traces,traces2))
        diagnoses = torch.cat((diagnoses,diagnoses2))
        traces, diagnoses = traces.to(device), diagnoses.to(device)  # use cuda if available

        # Reinitialize grad
        model.zero_grad()

        # Forward pass
        model_output = model(traces)
        loss = loss_function(model_output, diagnoses)

        # Backward pass
        loss.backward()

        # Optimize
        optimiser.step()

        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_batches += 1

        # Update progess bar
        train_bar.desc = train_desc.format(epoch=ep, loss=total_loss / n_batches)
        train_bar.update(1)
    train_bar.close()
    return total_loss/n_batches


def eval(ep, dataload, dataload2):
    model.eval()  # evaluation mode (e.g. dropout disabled)
    total_loss = 0  # accumulated loss

    # set up waitbar
    eval_desc = "Epoch {epoch:2d}: valid - Loss: {loss:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(epoch=ep, loss=0), position=0)

    # allocate space for storing the outputs
    eval_outputs = np.zeros(
        dataload.dset.in_chagas[dataload.start_idx:dataload.end_idx].sum()
        +dataload2.dset.in_chagas[dataload2.start_idx:dataload2.end_idx].sum())
    end = 0

    n_batches = 0 # accumulated number of batches

    # loop over the batches
    for nmbr, dload in enumerate([dataload, dataload2]):
        n_sub_batches = 0
        for traces, diagnoses in dload:
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
                n_batches += 1
                n_sub_batches += 1
                total_loss += loss.detach().cpu().numpy()
                display_loss = total_loss/n_batches

                # Print result
                eval_bar.desc = eval_desc.format(epoch=ep, loss=display_loss)
                eval_bar.update(1)

    total_loss /= n_batches

    eval_bar.close()
    return total_loss, eval_outputs


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict chagas from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum number of epochs (default: 100)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for number generator (default: 0)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                             'to fit into the given size. (default: 4096)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--batch_ratio', type=float, default=0.5,
                        help='proportion of samitrop dset in batch (default: 0.5).')
    parser.add_argument('--valid_split', type=float, default=0.15,
                        help='fraction of the data used for validation (samitrop) (default: 0.15).')
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
    parser.add_argument('--pos_weight', action='store_true',
                        help='use positive weighting in the loss. (default: False)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay in the optimisation. (default: 0)')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--path_to_data',
                        default='../data/samitrop/samitrop_all_records.h5',
                        help='path to file containing ECG traces for training (1st file)')
    parser.add_argument('--path_to_data2',
                        default='../data/code/traces.hdf5',
                        help='path to file containing ECG traces for training (2nd file)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file (1st file)')
    parser.add_argument('--traces_dset2', default='tracings',
                        help='traces dataset in the hdf5 file (2nd file)')
    parser.add_argument('--examid_dset', default='exam_id',
                        help='exam id dataset in the hdf5 file (1st file)')
    parser.add_argument('--examid_dset2', default='exam_id',
                        help='exam id dataset in the hdf5 file (2nd file)')
    parser.add_argument('--path_to_chagas', default='../data/chagas_samitrop2054.csv',
                        help='path to csv file containing chagas diagnoses (1st file)')
    parser.add_argument('--path_to_chagas2', default='../data/chagas_no_samitrop.csv',
                        help='path to csv file containing chagas diagnoses (2nd file)')

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

    # define data sets
    dset = ECGDatasetH5(
        path=args.path_to_data,
        traces_dset=args.traces_dset,
        exam_id_dset=args.examid_dset,
        ids_dset=None,
        path_to_chagas=args.path_to_chagas
    )
    dset2 = ECGDatasetH5(
        path=args.path_to_data2,
        traces_dset=args.traces_dset2,
        exam_id_dset=args.examid_dset2,
        ids_dset=None,
        path_to_chagas=args.path_to_chagas2
    )

    train_end = len(dset)
    n_valid = round(args.valid_split * train_end)
    n_valid2 = 2*n_valid
    train_end2 = round(args.data_tot * len(dset2))
    batch_size = round(args.batch_size*args.batch_ratio)
    batch_size2 = args.batch_size - batch_size

    valid_loader = ECGDataloaderH5(dset, args.batch_size, start_idx=0, end_idx=n_valid)
    train_loader = ECGDataloaderH5(dset, batch_size, start_idx=n_valid, end_idx=train_end)

    valid_loader2 = ECGDataloaderH5(dset2, args.batch_size, start_idx=0, end_idx=n_valid2)
    train_loader2 = ECGDataloaderH5(dset2, batch_size2, start_idx=n_valid2,
                                   end_idx=train_end2, cont_from_start=True)

    # true train & validation labels (to be used when computing metrics)
    train_true = np.concatenate((train_loader.getfullbatch(attr_only=True),
                                 train_loader2.getfullbatch(attr_only=True)))
    valid_true = np.concatenate((valid_loader.getfullbatch(attr_only=True),
                                 valid_loader2.getfullbatch(attr_only=True)))

    # save some data info
    n_train_chagas = train_true.size
    n_train_pos = train_true.sum()
    n_valid_chagas = valid_true.size
    n_valid_pos = valid_true.sum()
    data_info = 'n_total: {}\n\nn_train: {}\nn_train_pos: {}\nn_train_pos/n_train: {}' \
                '\n\nn_valid: {}\nn_valid_pos: {}\nn_valid_pos/n_valid: {}'\
                .format(n_valid_chagas+n_train_chagas, n_train_chagas, n_train_pos, n_train_pos/n_train_chagas,
                        n_valid_chagas, n_valid_pos, n_valid_pos/n_valid_chagas)
    file = open(os.path.join(args.folder, 'data_info.txt'), 'w+')
    file.write(data_info)
    file.close()

    # =============== Define model =============================================#
    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1  # two classes, but just need one output before sigmoid
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),  # (12, 4096)
                     blocks_dim=list(
                         zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
    model.to(device=device)
    tqdm.write("Done!")

    # =============== Define loss function =====================================#
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=None, reduction='mean')

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

    # create data frames to store the results in
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                        'valid_roc_auc', 'valid_avg_prec'])
    best_valid_output = pd.DataFrame(columns=['valid_true', 'valid_output'])
    best_valid_output['valid_true'] = valid_true

    # loop over epochs
    for ep in range(start_epoch, args.epochs):
        # compute losses and outputs for training & validation data
        train_loss = train(ep, train_loader, train_loader2)
        valid_loss, valid_outputs = eval(ep, valid_loader, valid_loader2)

        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),  # model state
                        'valid_loss': valid_loss,
                        'optimiser': optimiser.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss

            # save outputs
            best_valid_output['valid_output'] = valid_outputs
            best_valid_output.to_csv(os.path.join(folder, 'best_valid_output.csv'), index=False)

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
        # valid
        valid_roc_auc, valid_avg_prec = \
            compute_metrics(valid_true.astype(int), valid_outputs)[-2:]

        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                  "valid_loss": valid_loss,
                                  "lr": learning_rate,
                                  "valid_roc_auc": valid_roc_auc,
                                  "valid_avg_prec": valid_avg_prec},
                                 ignore_index=True)  # can only append a dict if ignore_index=True
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)

        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")
