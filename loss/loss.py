import torch
import torch.nn.functional as F

def orthogonality_loss(speaker_embedding, emotion_embedding):
    """
    Computes the orthogonality loss to enforce independence between speaker and emotion embeddings.
    """
    dot_product = torch.sum(speaker_embedding * emotion_embedding, dim=-1)
    speaker_norm = torch.norm(speaker_embedding, p=2, dim=-1)
    emotion_norm = torch.norm(emotion_embedding, p=2, dim=-1)
    cosine_similarity = dot_product / (speaker_norm * emotion_norm + 1e-8)
    loss = torch.mean(cosine_similarity ** 2)
    return loss

def ge2e_loss(embeddings, labels, margin=0.2):
    """
    Computes the Generalized End-to-End (GE2E) loss for speaker verification.
    """
    centroids = torch.zeros_like(embeddings)
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        mask = labels == label
        centroids[mask] = torch.mean(embeddings[mask], dim=0, keepdim=True)
    cosine_sim = F.cosine_similarity(embeddings, centroids, dim=-1)
    loss = torch.mean(F.relu(margin - cosine_sim))
    return loss

def mse_loss(predictions, targets):
    """
    Computes the Mean Squared Error (MSE) loss.
    """
    return torch.mean((predictions - targets) ** 2)

def classification_loss(logits, labels):
    """
    Computes the classification loss using Cross-Entropy.
    """
    return F.cross_entropy(logits, labels)
