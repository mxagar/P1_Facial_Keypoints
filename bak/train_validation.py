import time

# Training function
def train_net(n_epochs, device):
        
    # Transfer model/network to device
    net.to(device, dtype=torch.float)
    
    # prepare the net for training
    net.train()

    # metrics
    loss_history = {'training': [], 'validation': []}
    
    # Initialize min validation loss, for best model detection
    validation_loss_min = np.Inf
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        training_loss = 0.0
        running_loss = 0.0
        print_frequency = 30
        duration = 0.0
        num_samples = 0
        total_num_examples = 0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # Start time counter
            t1 = time.time()
            
            # Just in case
            net.train()
            
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # Transfer to device: cpu / cuda
            images, key_pts = images.to(device, dtype=torch.float), key_pts.to(device, dtype=torch.float)
            
            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)                
                
            # forward pass to get outputs
            output_pts = net(images)
            
            # flatten outputs
            #output_pts = output_pts.view(output_pts.size(0), -1)
            
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            t2 = time.time()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            training_loss += loss.item()
            num_samples += images.size(0)
            total_num_examples += images.size(0)
            duration += t2-t1
            
            if batch_i % print_frequency == 9:    # print every print_frequency batches
                print(f'Epoch-Batch: {epoch + 1}-{batch_i + 1}, Train Loss: {np.round(running_loss/num_samples,6)}, Batch duration: {np.round(duration/print_frequency,2)} sec.')
                running_loss = 0.0
                num_samples = 0
                duration = 0.0
        
        # When an epoch finishes
        
        # Compute validation and traning losses for the epoch
        #training_loss /= float(len(train_loader))
        training_loss /= total_num_examples
        validation_loss = validate(net, criterion, test_loader, device)

        # Store losses
        loss_history['training'].append(training_loss)
        loss_history['validation'].append(validation_loss)

        # Print
        print(f'Epoch: {epoch + 1}, Training Loss: {training_loss}, Validation Loss: {validation_loss}')

        # Save always last model
        filepath = 'saved_models/model_last.pth'
        save_model(filepath, net, epoch, validation_loss, loss_history)

        # Save model with best validation error
        if validation_loss < validation_loss_min:
            validation_loss_min = validation_loss
            filepath = 'saved_models/model_best.pth'
            save_model(filepath, net, epoch, validation_loss, loss_history)

        # We could implement early stopping
        # by observing the loss_history
        # TODO
            
    print('Finished Training!')
    return loss_history
