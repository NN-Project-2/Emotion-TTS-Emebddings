import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, dataloader, num_epochs, learning_rate, checkpoint_dir, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)  
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, language_ids, emotion_ids in dataloader:
            features = features.to(device)
            
            optimizer.zero_grad()
            
            latent, _, reconstructed = model(features)
            loss = criterion(reconstructed, features)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
