from sklearn.base import defaultdict
import torch
import torch.nn.functional as F
import os


import json
import os
import matplotlib.pyplot as plt
import pandas as pd


import json
import matplotlib.pyplot as plt
import pandas as pd


import json
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn.functional as F

def _canon_key(k: str) -> str:
    # strip dataset prefix like "flickr30k-val/..."
    if "/" in k:
        k = k.split("/", 1)[1]

    # harmonize names
    k = k.replace("text_to_image_R@", "txt_r")
    k = k.replace("image_to_text_R@", "img_r")
    k = k.replace("text_to_image_mean_rank", "txt_mean_rank")
    k = k.replace("text_to_image_median_rank", "txt_median_rank")
    k = k.replace("image_to_text_mean_rank", "img_mean_rank")
    k = k.replace("image_to_text_median_rank", "img_median_rank")
    k = k.replace("@", "_at_")
    return k


def extract_and_plot_itm_scores(
    results_file_path,
    output_plot_path,
    output_similarity_plot_path,
    save_csv_path=None,
    prefer_dataset=None,
    output_modality_gap_plot_path=None,  # NEW (optional)
):
    def _mean_of(keys, d):
        vals = [d[k] for k in keys if k in d]
        return sum(vals) / len(vals) if vals else 0.0

    epochs = []
    txt_r1, txt_r5, txt_r10, txt_r_mean = [], [], [], []
    img_r1, img_r5, img_r10, img_r_mean = [], [], [], []
    r_mean = []
    average_similarity = []
    modality_gap = []  # NEW

    with open(results_file_path, "r") as f:
        results = json.load(f)

    for entry in results:
        epoch = entry.get("epoch", None)
        if epoch is None:
            continue
        res_list = entry.get("results", [])
        if not res_list:
            continue

        # pick the dataset to plot; default = first
        chosen = None
        if prefer_dataset is not None:
            for r in res_list:
                if r.get("val_name") == prefer_dataset:
                    chosen = r
                    break
        if chosen is None:
            chosen = res_list[0]

        # tolerate either nested {"metrics": {...}} or flat dicts
        metrics_raw = chosen.get("metrics", chosen)

        # normalize keys
        m = {}
        for k, v in metrics_raw.items():
            if isinstance(v, (int, float)):
                m[_canon_key(k)] = float(v)

        # fill derived means if missing
        if "txt_r_mean" not in m:
            m["txt_r_mean"] = _mean_of(["txt_r1", "txt_r5", "txt_r10"], m)
        if "img_r_mean" not in m:
            m["img_r_mean"] = _mean_of(["img_r1", "img_r5", "img_r10"], m)
        if "r_mean" not in m:
            if "txt_r_mean" in m and "img_r_mean" in m:
                m["r_mean"] = (m["txt_r_mean"] + m["img_r_mean"]) / 2.0
            else:
                m["r_mean"] = _mean_of(
                    ["txt_r1", "txt_r5", "txt_r10", "img_r1", "img_r5", "img_r10"], m
                )

        epochs.append(epoch)
        txt_r1.append(m.get("txt_r1", 0.0))
        txt_r5.append(m.get("txt_r5", 0.0))
        txt_r10.append(m.get("txt_r10", 0.0))
        txt_r_mean.append(m.get("txt_r_mean", 0.0))
        img_r1.append(m.get("img_r1", 0.0))
        img_r5.append(m.get("img_r5", 0.0))
        img_r10.append(m.get("img_r10", 0.0))
        img_r_mean.append(m.get("img_r_mean", 0.0))
        r_mean.append(m.get("r_mean", 0.0))
        average_similarity.append(m.get("average_similarity", 0.0))
        modality_gap.append(m.get("modality_gap", 0.0))  # NEW

    if not epochs:
        raise ValueError(f"No usable results found in {results_file_path}")

    # 🏆 Best & final R_mean
    best_r_mean = max(r_mean)
    best_r_mean_epoch = epochs[r_mean.index(best_r_mean)]
    final_r_mean = r_mean[-1]
    final_epoch = epochs[-1]

    # Plot recall scores
    plt.figure()
    plt.plot(epochs, txt_r1, label="Text R@1")
    plt.plot(epochs, txt_r5, label="Text R@5")
    plt.plot(epochs, txt_r10, label="Text R@10")
    plt.plot(epochs, img_r1, label="Image R@1")
    plt.plot(epochs, img_r5, label="Image R@5")
    plt.plot(epochs, img_r10, label="Image R@10")
    plt.plot(epochs, r_mean, label="R Mean", linestyle="--", linewidth=2)
    plt.scatter([best_r_mean_epoch], [best_r_mean], label=f"Best R Mean ({best_r_mean:.2f})")
    plt.scatter([final_epoch], [final_r_mean], label=f"Final R Mean ({final_r_mean:.2f})")
    plt.xlabel("Epoch"); plt.ylabel("Recall (%)"); plt.title("ITM Recall Scores over Epochs")
    plt.legend(); plt.grid(True); plt.savefig(output_plot_path); plt.close()

    # Best & final average similarity (higher is better)
    best_sim = max(average_similarity)
    best_sim_epoch = epochs[average_similarity.index(best_sim)]
    final_sim = average_similarity[-1]

    plt.figure()
    plt.plot(epochs, average_similarity, label="Average Similarity", linestyle=":", linewidth=2)
    plt.scatter([best_sim_epoch], [best_sim], label=f"Best Sim ({best_sim:.4f})")
    plt.scatter([final_epoch], [final_sim], label=f"Final Sim ({final_sim:.4f})")
    plt.xlabel("Epoch"); plt.ylabel("Similarity"); plt.title("Average Similarity over Epochs")
    plt.legend(); plt.grid(True); plt.savefig(output_similarity_plot_path); plt.close()

    # NEW: modality gap plot (lower is better)
    if output_modality_gap_plot_path is None and output_similarity_plot_path:
        base_dir = os.path.dirname(output_similarity_plot_path)
        output_modality_gap_plot_path = os.path.join(base_dir, "modality_gap_plot.png")

    best_gap = min(modality_gap)
    best_gap_epoch = epochs[modality_gap.index(best_gap)]
    final_gap = modality_gap[-1]

    plt.figure()
    plt.plot(epochs, modality_gap, label="Modality Gap (L2)", linewidth=2)
    plt.scatter([best_gap_epoch], [best_gap], label=f"Best (min) Gap ({best_gap:.4f})")
    plt.scatter([final_epoch], [final_gap], label=f"Final Gap ({final_gap:.4f})")
    plt.xlabel("Epoch"); plt.ylabel("L2 Distance")
    plt.title("Modality Gap over Epochs")
    plt.legend(); plt.grid(True); plt.savefig(output_modality_gap_plot_path); plt.close()

    if save_csv_path:
        df = pd.DataFrame({
            "epoch": epochs,
            "txt_r1": txt_r1, "txt_r5": txt_r5, "txt_r10": txt_r10, "txt_r_mean": txt_r_mean,
            "img_r1": img_r1, "img_r5": img_r5, "img_r10": img_r10, "img_r_mean": img_r_mean,
            "r_mean": r_mean, "average_similarity": average_similarity,
            "modality_gap": modality_gap,  # NEW
        })
        df.to_csv(save_csv_path, index=False)

    # console summary
    print("\n=== ITM Metric Arrays ===")
    print("Epochs:", epochs)
    print("Text R@1:", txt_r1); print("Text R@5:", txt_r5); print("Text R@10:", txt_r10)
    print("Image R@1:", img_r1); print("Image R@5:", img_r5); print("Image R@10:", img_r10)
    print("R Mean:", r_mean)
    print("Average Similarity:", average_similarity)
    print("Modality Gap:", modality_gap)  # NEW

    best_idx = r_mean.index(best_r_mean)
    best_epoch_metrics = {
        "epoch": epochs[best_idx],
        "txt_r1": txt_r1[best_idx], "txt_r5": txt_r5[best_idx], "txt_r10": txt_r10[best_idx],
        "img_r1": img_r1[best_idx], "img_r5": img_r5[best_idx], "img_r10": img_r10[best_idx],
        "r_mean": r_mean[best_idx],
        "average_similarity": average_similarity[best_idx],
        "modality_gap": modality_gap[best_idx],  # NEW
    }
    print("\n=== Best Epoch Metrics (by R Mean) ===")
    print(", ".join(f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}"
                    for k, v in best_epoch_metrics.items()))

    return {
        "best_r_mean": (best_r_mean_epoch, best_r_mean),
        "final_r_mean": (final_epoch, final_r_mean),
        "best_similarity": (best_sim_epoch, best_sim),
        "final_similarity": (final_epoch, final_sim),
        "best_modality_gap": (best_gap_epoch, best_gap),    # NEW (min)
        "final_modality_gap": (final_epoch, final_gap),     # NEW
    }

def compute_consistency_score(model, dataloader, device):
    """
    Computes the Consistency Score for a given model and dataset.

    Args:
        model: The trained CyCLIP model.
        dataloader: DataLoader providing (image, text) pairs.
        device: Device to run the computations on ('cuda' or 'cpu').

    Returns:
        The average Consistency Score across the dataset.
    """
    model.eval()
    total_score = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, texts in dataloader:
            images, texts = images.to(device), texts.to(device)

            # Encode images and texts
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)

            # Compute cosine similarity
            similarities = torch.sum(image_features * text_features, dim=-1)

            # Accumulate scores
            total_score += similarities.sum().item()
            num_samples += images.size(0)

    # Average consistency score
    average_score = total_score / num_samples
    return average_score


#       USAGE EXAMPLE

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# consistency_score = compute_consistency_score(model, dataloader, device)
# print(f"Consistency Score: {consistency_score:.4f}")


import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(
    model,
    preprocess,
    tokenizer,
    test_data,
    image_folder,
    device,
    top_k=5,
    captions_per_image=5,
):
    """
    Evaluates an image-text retrieval model using a preloaded test dataset.

    Args:
        model: The OpenCLIP model to use for encoding.
        preprocess: The preprocessing function for images.
        tokenizer: The tokenizer for text encoding.
        test_data (list): Preloaded test dataset (list of image-caption dictionaries).
        image_folder (str): Path to the folder containing Flickr30k images.
        device: PyTorch device (e.g., "cuda" or "cpu").
        top_k (int): Number of top retrieved captions/images to consider.
        captions_per_image (int): Number of captions associated with each image.

    Returns:
        DataFrame containing image paths, retrieved captions, and similarity scores.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store extracted features
    image_features = []
    text_features = []
    image_paths = []
    captions = []
    caption_to_image = {}

    # Process each test image
    for item in tqdm(test_data, desc="Extracting Features"):
        img_path = f"{image_folder}/{item['filename']}"
        image_paths.append(img_path)

        # Encode image
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(image)
        image_features.append(img_feat.cpu().numpy())

        # Encode captions
        for sentence in item["sentences"]:
            text = sentence["raw"]
            text_tokenized = tokenizer([text]).to(device)

            with torch.no_grad():
                text_feat = model.encode_text(text_tokenized)

            text_features.append(text_feat.cpu().numpy())
            captions.append(text)
            caption_to_image[text] = (
                img_path  # Store which image the caption belongs to
            )

    # Convert to numpy arrays
    image_features = np.vstack(image_features)
    text_features = np.vstack(text_features)

    print(
        f"✅ Extracted {image_features.shape[0]} image features and {text_features.shape[0]} text features."
    )

    # Normalize features
    image_features = image_features / np.linalg.norm(
        image_features, axis=1, keepdims=True
    )
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

    # Compute similarity matrix (dot product)
    similarity_matrix = np.dot(text_features, image_features.T)

    # Compute retrieval accuracy
    def evaluate_retrieval(similarity_matrix, top_k=1, captions_per_image=5):
        N_captions, N_images = similarity_matrix.shape
        assert (
            N_captions == N_images * captions_per_image
        ), f"Number of captions ({N_captions}) must be {captions_per_image} times the number of images ({N_images})."

        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k:]
        correct = 0
        for caption_idx in range(N_captions):
            correct_image_idx = caption_idx // captions_per_image
            if correct_image_idx in top_k_indices[caption_idx]:
                correct += 1
        return correct / N_captions

    # Compute accuracy for top-1, top-5, top-10
    for k in [1, 5, 10]:
        acc = evaluate_retrieval(
            similarity_matrix, top_k=k, captions_per_image=captions_per_image
        )
        print(f"Top-{k} Accuracy: {acc * 100:.2f}%")

    # Retrieve top-k captions per image
    top_k_indices = np.argsort(similarity_matrix, axis=0)[
        -top_k:
    ].T  # Sorting and taking top-k

    # Store results
    results = []
    for img_idx, indices in enumerate(top_k_indices):
        img_path = image_paths[img_idx]
        retrieved_captions = [captions[idx] for idx in indices[::-1]]
        retrieved_probs = [similarity_matrix[idx, img_idx] for idx in indices[::-1]]

        results.append(
            {
                "Image": img_path,
                "Top-5 Matches": list(zip(retrieved_captions, retrieved_probs)),
            }
        )

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    return df_results


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
#                               FROM CYCLIP
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


# @torch.no_grad()
# def flickr_retrieval_eval_(text_embeddings, image_embeddings):
    device = text_embeddings.device
    N = len(image_embeddings)
    
    # Normalize features
    image_features = F.normalize(image_embeddings, p=2, dim=-1)
    text_features = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Compute average similarity
    average_similarity = torch.sum(image_features * text_features, dim=-1).mean().item()

    # Image → Text retrieval (assuming every 5 captions = 1 image)
    ranks = torch.zeros(N, dtype=torch.int32, device=device)
    for index in range(0, N, 5):
        scores = image_embeddings[index] @ text_embeddings.T
        sorted_indices = torch.argsort(scores, descending=True)
        for rank, idx in enumerate(sorted_indices):
            if index <= idx < index + 5:  # Correct caption is in this image's group
                ranks[index] = rank
                break

    tr1 = 100.0 * torch.sum(ranks < 1).item() / N
    tr5 = 100.0 * torch.sum(ranks < 5).item() / N
    tr10 = 100.0 * torch.sum(ranks < 10).item() / N

    # Text → Image retrieval
    ranks = torch.zeros(N, dtype=torch.int32, device=device)
    for index in range(N):
        scores = text_embeddings[index] @ image_embeddings.T
        # Only compare with image features (every 5th element)
        image_scores = scores[::5]  
        sorted_indices = torch.argsort(image_scores, descending=True)
        target_img_idx = index // 5
        for rank, img_idx in enumerate(sorted_indices):
            if img_idx == target_img_idx:
                ranks[index] = rank
                break

    ir1 = 100.0 * torch.sum(ranks < 1).item() / N
    ir5 = 100.0 * torch.sum(ranks < 5).item() / N
    ir10 = 100.0 * torch.sum(ranks < 10).item() / N

    return {
        "txt_r1": tr1, "txt_r5": tr5, "txt_r10": tr10,
        "img_r1": ir1, "img_r5": ir5, "img_r10": ir10,
        "average_similarity": average_similarity,
    }
#OK NUMBERS ABOVE

import torch
import torch.nn.functional as F

@torch.no_grad()
def flickr_retrieval_eval_(text_embeddings, image_embeddings):
    device = text_embeddings.device
    N = text_embeddings.shape[0]
    assert N % 5 == 0
    n_img = N // 5

    I = F.normalize(image_embeddings, p=2, dim=-1)
    T = F.normalize(text_embeddings,  p=2, dim=-1)

    average_similarity = torch.sum(I * T, dim=-1).mean().item()

    # --- Image → Text (per image) ---
    ranks_img = torch.empty(n_img, dtype=torch.int32, device=device)
    for g in range(n_img):
        img_row = g * 5                  # use the FIRST occurrence (don’t average)
        scores = I[img_row] @ T.t()
        sorted_idx = torch.argsort(scores, descending=True)
        # best (min) rank among this image's 5 captions
        best = torch.iinfo(torch.int32).max
        for j in range(img_row, img_row + 5):
            pos = (sorted_idx == j).nonzero(as_tuple=True)[0][0].item()
            if pos < best:
                best = pos
        ranks_img[g] = best

    tr1  = 100.0 * (ranks_img < 1).float().mean().item()
    tr5  = 100.0 * (ranks_img < 5).float().mean().item()
    tr10 = 100.0 * (ranks_img < 10).float().mean().item()

    # --- Text → Image (per caption) ---
    # compare against images at rows 0,5,10,... (first occurrence per image)
    img_rows = torch.arange(0, N, 5, device=device)  # length n_img
    ranks_cap = torch.empty(N, dtype=torch.int32, device=device)
    for i in range(N):
        scores = T[i] @ I[img_rows].t()              # [n_img]
        sorted_idx = torch.argsort(scores, descending=True)
        target = i // 5                               # image index for this caption
        pos = (sorted_idx == target).nonzero(as_tuple=True)[0][0].item()
        ranks_cap[i] = pos

    ir1  = 100.0 * (ranks_cap < 1).float().mean().item()
    ir5  = 100.0 * (ranks_cap < 5).float().mean().item()
    ir10 = 100.0 * (ranks_cap < 10).float().mean().item()

    return {
        "txt_r1": tr1, "txt_r5": tr5, "txt_r10": tr10,
        "img_r1": ir1, "img_r5": ir5, "img_r10": ir10,
        "average_similarity": average_similarity,
    }
def get_all_embeddings(
    model,
    all_texts,
    all_images,
    root,
    preprocess,
    tokenizer,
    batch_size=1024,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_embeddings = []
    image_embeddings = []

    with torch.no_grad():
        dataloader_texts = list(batch(all_texts, batch_size))
        dataloader_images = list(batch(all_images, batch_size))

        bar = zip(dataloader_texts, dataloader_images)
        bar = tqdm(bar, total=len(dataloader_texts), desc="Encoding batches")

        for texts, images in bar:
            # Tokenize text
            text_tokens = tokenizer(texts).to(device)

            # Preprocess and stack images
            image_tensors = torch.stack(
                [
                    preprocess(Image.open(os.path.join(root, img)).convert("RGB"))
                    for img in images
                ]
            ).to(device)

            # Encode
            image_embedding = model.encode_image(image_tensors)
            text_embedding = model.encode_text(text_tokens)

            # Normalize
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

            text_embeddings.append(text_embedding)
            image_embeddings.append(image_embedding)

        text_embeddings = torch.cat(text_embeddings)
        image_embeddings = torch.cat(image_embeddings)
        return text_embeddings, image_embeddings


from tqdm import tqdm
from PIL import Image
import os
import torch


# 2. Image embedding function
def load_image_embeddings(model, preprocess, unique_images, image_folder, device):
    model.eval()
    image_embeddings = []

    for filename in tqdm(unique_images, desc="Encoding images"):
        image_path = f"{image_folder}/{filename}"
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            image_embeddings.append(image_feature)

    return torch.cat(image_embeddings, dim=0)  # shape: [N, D]


# 3. Text embedding function
def load_text_embeddings(all_captions, model, tokenizer, device):
    model.eval()
    text_embeddings = []

    for caption in tqdm(all_captions, desc="Encoding captions"):
        text_tokens = tokenizer([caption]).to(device)  # batch of size 1
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_embeddings.append(text_feature)

    return torch.cat(text_embeddings, dim=0)  # shape: [5N, D]
