import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model.EmotionModel import EmotionModel


class GE2ELoss(nn.Module):
    def __init__(self, margin=0.2):
        super(GE2ELoss, self).__init__()
        self.margin = margin

    def forward(self, speaker_embeddings):
        similarity_matrix = F.cosine_similarity(speaker_embeddings.unsqueeze(1), speaker_embeddings.unsqueeze(0), dim=-1)
        identity_mask = torch.eye(similarity_matrix.size(0)).to(speaker_embeddings.device)
        similarity_matrix = similarity_matrix * (1 - identity_mask)
        loss = torch.clamp(self.margin - similarity_matrix, min=0.0).mean()
        return loss


class OrthogonalityLoss(nn.Module):
    def forward(self, speaker_embedding, emotion_embedding):
        similarity = F.cosine_similarity(speaker_embedding, emotion_embedding, dim=-1)
        loss = torch.mean(torch.abs(similarity))
        return loss


class EmotionClassificationLoss(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassificationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, emotion_embedding, emotion_labels):
        loss = self.cross_entropy(emotion_embedding, emotion_labels)
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel_reconstructed, mel_refined, ground_truth):
        loss_recon = self.l1_loss(mel_reconstructed, ground_truth)
        loss_refine = self.l1_loss(mel_refined, ground_truth)
        return loss_recon + loss_refine


class Trainer:
    def __init__(self, model, dataloader, num_epochs, learning_rate, checkpoint_dir, device, num_emotions):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.reconstruction_loss = ReconstructionLoss()
        self.ge2e_loss = GE2ELoss(margin=0.2)
        self.orthogonality_loss = OrthogonalityLoss()
        self.emotion_classification_loss = EmotionClassificationLoss(num_classes=num_emotions)

        # Optimizer and Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.7)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def calculate_losses(self, features, labels):
        
        mel_reconstructed, mel_refined, speaker_embedding, emotion_embedding, pred_emotion = self.model(features)

        loss_recon = self.reconstruction_loss(mel_reconstructed, mel_refined, features)
        loss_speaker = self.ge2e_loss(speaker_embedding)
        loss_ortho = self.orthogonality_loss(speaker_embedding, emotion_embedding)
        loss_emotion = self.emotion_classification_loss(pred_emotion, labels)


        total_loss = (
            1.0 * loss_recon +        # Mel Reconstruction Loss
            0.5 * loss_speaker +      # Speaker Disentanglement Loss
            0.5 * loss_ortho +        # Orthogonality Loss
            1.0 * loss_emotion        # Emotion Classification Loss
        )

        return total_loss, {
            "Reconstruction Loss": loss_recon.item(),
            "Speaker Loss (GE2E)": loss_speaker.item(),
            "Orthogonality Loss": loss_ortho.item(),
            "Emotion Classification Loss": loss_emotion.item()
        }

    def train(self):
        best_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            losses = {"Reconstruction Loss": 0.0,
                      "Speaker Loss (GE2E)": 0.0,
                      "Orthogonality Loss": 0.0,
                      "Emotion Classification Loss": 0.0}

            for features, labels, _ in self.dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                total_loss, loss_dict = self.calculate_losses(features, labels)

                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()
                for k, v in loss_dict.items():
                    losses[k] += v

            avg_loss = running_loss / len(self.dataloader)
            avg_loss_dict = {k: v / len(self.dataloader) for k, v in losses.items()}

            print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {avg_loss:.4f}")
            for k, v in avg_loss_dict.items():
                print(f"    {k}: {v:.4f}")


            self.scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, best_model=True)
                
            if epoch % 100 == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch, best_model=False):
        filename = "best_model.pth" if best_model else f"model_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
