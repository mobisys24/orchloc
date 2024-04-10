import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import os
import torch


from src.dataset import generate_three_loader_v3
from src.parameter_paser import parse_args_area_2
from src.autoencoder import LocationClassifier


def loss_function(label_batch, label_pred):
    return nn.CrossEntropyLoss()(label_pred, label_batch)


def training(model,
             learning_rate_t, 
             dataloader_t, 
             valid_data_loader_t, 
             model_path_t, 
             num_epochs_t, 
             device):
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_t)  # AdamW, Adam, RMSprop
    
    print('\nModel training...\n')

    train_loss = []
    valid_loss = []
    valid_acc = []
    for i_epoch in range(num_epochs_t):
        model.train()  # Set the model to training mode
        total_loss_this_epoch = 0
        for batch_idx, (data_batch_fake, _, _, label_int_batch) in enumerate(dataloader_t):
            
            data_batch_fake = data_batch_fake.to(device)
            label_int_batch = label_int_batch.to(device)

            optimizer.zero_grad()

            label_pred_onehot = model(data_batch_fake)
            loss = loss_function(label_int_batch, label_pred_onehot)
            loss.backward()
            optimizer.step()
            total_loss_this_epoch += loss.item()

            if batch_idx % 10 == 0:
                print('--------> Epoch: {}/{} loss: {:.8f} [{}/{} ({:.0f}%)]'.format(i_epoch + 1,
                                                                                      num_epochs_t, 
                                                                                      loss.item() / len(data_batch_fake), 
                                                                                      batch_idx * len(data_batch_fake), 
                                                                                      len(dataloader_t.dataset), 
                                                                                      100.0 * batch_idx / len(dataloader_t)
                                                                                      ), end='\r')

        model.eval()
        correct_count = total_count = 0
        total_valid_loss = 0
        with torch.no_grad():
            for batch_idx, (_, data_batch_real, _, label_int_batch) in enumerate(valid_data_loader_t):

                data_batch_real = data_batch_real.to(device)

                label_pred_onehot = model(data_batch_real).cpu()             # [@, n_class]
                label_pred = torch.argmax(label_pred_onehot, dim=-1)    # [@]

                correct_count += (label_pred == label_int_batch).sum().item()
                total_count += label_pred_onehot.shape[0]
                total_valid_loss += loss_function(label_int_batch, label_pred_onehot).item()

        print('========> Epoch: {}/{} Loss: {:.8f}, valid accuracy: {:.3f}%({}/{})'.format(i_epoch + 1, 
                                                                                           num_epochs_t, 
                                                                                           total_loss_this_epoch / len(dataloader_t.dataset), 
                                                                                           100.0 * correct_count / total_count, 
                                                                                           correct_count, 
                                                                                           total_count
                                                                                           ) + ' ' * 20)
        
        train_loss.append(total_loss_this_epoch / len(dataloader_t.dataset))
        valid_loss.append(total_valid_loss / len(valid_data_loader_t.dataset))
        valid_acc.append(100.0 * correct_count / total_count)

    torch.save(model.state_dict(), model_path_t)

    print('\nTraining finished\n')

    return train_loss, valid_loss, valid_acc


def testing(model, 
            dataloader_t, 
            device):
    
    print('\nModel testing...')
    print("Loaded classifier model: {}\n".format(model_path_train))

    correct_count = 0
    total_count = 0

    # turn off the gradients computation
    model.eval()
    with torch.no_grad():

        # only one iteration
        for _, (_, data_batch_real, _, label_int_batch) in enumerate(dataloader_t):  # for all test samples

        
            data_batch_real = data_batch_real.to(device)

            label_pred_onehot = model(data_batch_real).cpu()         # [@, n_class]
            label_pred = torch.argmax(label_pred_onehot, dim=-1)     # [@]

            correct_count += (label_pred == label_int_batch).sum().item()
            total_count += label_pred_onehot.shape[0]

            acc = (correct_count * 1.0) / (total_count * 1.0) * 100
            precision = precision_score(label_int_batch, label_pred, average='weighted') * 100
            recall = recall_score(label_int_batch, label_pred, average='weighted') * 100
   
            print('Test accuracy: {:.6f}% ({}/{}), precision: {:.6f}%, recall: {:.6f}%\n'.format(
                acc, 
                correct_count,
                total_count,
                precision, 
                recall))


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))

    args = parse_args_area_2(base_dir)
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = os.path.join(base_dir, "input")

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    model_path_train = os.path.join(output_dir, args.save_model_name_fake)

    data_path_area_fake = data_path = os.path.join(output_dir, args.data_name_fake)

    num_locs = args.num_locs
    train_flag = args.train_flag

    frac_for_valid = args.frac_for_valid
    frac_for_test = args.frac_for_test

    # Parameters for location classifier
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    dropout_pra = args.dropout_pra
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size_t = args.batch_size

    complex_dataset_generated_real = torch.load(data_path_area_fake)

    train_loader, valid_loader, test_loader = generate_three_loader_v3(complex_dataset_generated_real, 
                                                            batch_size_t, 
                                                            frac_for_valid, 
                                                            frac_for_test)
    
    device_m = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier_model = LocationClassifier(input_dim, 
                                            hidden_dim, 
                                            n_layers, 
                                            dropout_pra, 
                                            num_locs).to(device_m)

    if train_flag:
        # using generated data to train
        training(classifier_model, 
                learning_rate, 
                train_loader, 
                valid_loader, 
                model_path_train, 
                num_epochs, 
                device_m)
    
    ##### Testing #####
    # use the real collected data to test
    classifier_model.load_state_dict(torch.load(model_path_train))

    testing(classifier_model, 
            test_loader, 
            device_m)
    