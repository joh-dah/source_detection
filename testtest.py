import torch


def subsampleClasses(y: torch.Tensor, y_hat: torch.Tensor):
    # subsample the majority class
    non_sources = torch.where(y == 0)[0]
    sources = torch.where(y == 1)[0]
    random_numbers = torch.randperm(non_sources.shape[0])[: sources.shape[0]]
    subsampled_non_sources = non_sources[random_numbers]
    indices = torch.cat((subsampled_non_sources, sources))
    return y[indices], y_hat[indices]


if __name__ == "__main__":
    y = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    y_hat = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1])
    y, y_hat = subsampleClasses(y, y_hat)
    print(y)
    print(y_hat)
