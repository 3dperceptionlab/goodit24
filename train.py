"""
Train birds behaviors model on Chantwin dataset.

Usage:
    train.py [options] [-h]

Options:
    --model=<model>             Model to use. Possible values: 3dresnet|swin|s3d|mvit|r2plus [default: 3dresnet]
    --checkpoint_path=<checkpoint_path>     Path to a checkpoint to load the model from
    --data_path=<data_path>     Dataset to use [default: /data/images]
    --annot_path=<annot_path>   Annotations path [default: /data/annotations.csv]
    --batch_size=<batch_size>   Batch size [default: 8]
    --epochs=<epochs>           Number of epochs [default: 100]
    --patience=<patience>       Patience for early stopping [default: 10]
    --eval                     Only evaluate the model
    --freeze                   Freeze the model
    -h --help                   Show this screen
"""

from docopt import docopt
from dataset import BehaviorsDataset
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights, s3d, S3D_Weights, swin3d_t, Swin3D_T_Weights, mvit_v1_b, MViT_V1_B_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
import torch
import time
from tqdm import tqdm
import wandb
import numpy as np

def main(args):
    # Load arguments
    data_path = args['--data_path']
    annot_path = args['--annot_path']
    batch_size = int(args['--batch_size'])
    epochs = int(args['--epochs'])
    checkpoint_path = args['--checkpoint_path']
    eval_only = args['--eval']
    patience = int(args['--patience'])
    model_name = args['--model']

    # Load dataset
    print('Loading datasets...')
    if not eval_only:
        train_dataset = BehaviorsDataset(annot_path, data_path, split='train', model_name=model_name)
        print('Train dataset loaded')
        val_dataset = BehaviorsDataset(annot_path, data_path, split='val', model_name=model_name)
        print('Validation dataset loaded')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    else:
        test_dataset = BehaviorsDataset(annot_path, data_path, split='test', model_name=model_name)
        print('Test dataset loaded')
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    # Load model
    if model_name == '3dresnet':
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 8)
    elif model_name == 'swin':
        weights = Swin3D_T_Weights.DEFAULT
        model = swin3d_t(weights=weights)
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, 8)
    elif model_name == 's3d':
        weights = S3D_Weights.DEFAULT
        model = s3d(weights=weights)
        in_features = model.classifier[1].in_channels
        model.classifier[1] = torch.nn.Conv3d(in_features, 8, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    elif model_name == 'mvit':
        weights = MViT_V1_B_Weights.DEFAULT
        model = mvit_v1_b(weights=weights)
        in_features = model.head[1].in_features
        model.head[1] = torch.nn.Linear(in_features, 8)
    elif model_name == 'r2plus':
        weights = R2Plus1D_18_Weights.DEFAULT
        model = r2plus1d_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 8)
    elif model_name == 'swin3d_b':
        weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        model = swin3d_b(weights=weights)
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, 8)
    else:
        raise ValueError('Model not recognized')
        
    ## TODO: LOAD WEIGHTS OF CHECKPOINT PATH
    if checkpoint_path != None:
        model.load_state_dict(torch.load(checkpoint_path))

    # Freeze model except classifier
    if args['--freeze']:
        for param in model.parameters():
            param.requires_grad = False
        if model_name == '3dresnet':
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == 'swin':
            for param in model.head.parameters():
                param.requires_grad = True
        elif model_name == 's3d':
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        elif model_name == 'r2plus':
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == 'mvit':
            for param in model.head[1].parameters():
                param.requires_grad = True
        elif model_name == 'swin3d_b':
            for param in model.head.parameters():
                param.requires_grad = True
    

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    max_val_accuracy = 0.0
    min_val_loss = float('inf')
    early_stop = False
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        # Early stopping
        if early_stop:
            print('Early stopping')
            break

        # Training
        print(f'Epoch {epoch+1}/{epochs}')
        start = time.time()
        if not eval_only:
            epoch_losses = []
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total = 0
            val_loss = []
            all_predictions = []
            all_labels = []

            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            val_accuracy = total_correct / total
            if not eval_only:
                if val_accuracy > max_val_accuracy:
                    max_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), f'weights/best_model_{model_name}.pth')
                
                if sum(val_loss)/len(val_loss) < min_val_loss:
                    min_val_loss = sum(val_loss)/len(val_loss)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        early_stop = True
            else:
                from sklearn.metrics import confusion_matrix
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                import matplotlib.pyplot as plt
                c_matrix = confusion_matrix(all_labels, all_predictions)
                disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=np.unique(all_labels))
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(f'data/confusion_matrix_{model_name}.png')
        end = time.time()

        # Print results
        if not eval_only:
            print(f'Epoch {epoch+1}/{epochs} - Loss: {sum(epoch_losses)/len(epoch_losses)} - Val loss: {sum(val_loss)/len(val_loss)} - Val accuracy: {val_accuracy} - Time: {end-start}')
        else:
            print(f'Val loss: {sum(val_loss)/len(val_loss)} - Val accuracy: {val_accuracy} - Time: {end-start}')
        if not eval_only:
            wandb.log({'epoch': epoch, 'train_loss': sum(epoch_losses)/len(epoch_losses), 'val_loss': sum(val_loss)/len(val_loss), 'val_accuracy': val_accuracy})
    wandb.finish()



if __name__ == '__main__':
    args = docopt(__doc__)
    #wandb.init(project='birds-behaviours', name=args['--model'])
    wandb.init(mode='disabled')
    main(args)