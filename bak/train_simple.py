import time

# Training function
def train_net(n_epochs, device):

    # Transfer model/network to device
    net.to(device, dtype=torch.float)

    # prepare the net for training
    net.train()

    # metrics
    loss_history = {'training': [], 'validation': []}

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        print_frequency = 10

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # Transfer to device: cpu / cuda
            images, key_pts = images.to(device, dtype=torch.float), key_pts.to(device, dtype=torch.float)

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            #key_pts = key_pts.type(torch.FloatTensor)
            #images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % print_frequency == 9:    # print every 10 batches
                loss_history['training'].append(running_loss/print_frequency)
                loss_history['validation'].append(running_loss/print_frequency) # TODO: Change or remove
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/print_frequency))
                running_loss = 0.0

        # After each epoch
        # Save always last model
        filepath = 'saved_models/model_last.pth'
        validation_loss = loss_history['training'][-1]
        save_model(filepath, net, epoch, validation_loss, loss_history)


    print('Finished Training')

    return loss_history
