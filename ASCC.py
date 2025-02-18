import cupy as cp
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def compute_fft_correlation(signal1, signal2):
    """
    Use FFT to calculate the normalized cross correlation of signal 1 and signal 2
    Returns the normalized cross-correlation maximum as the correlation coefficient of the two signals
    """
    f1 = cp.fft.fft(signal1)
    f2 = cp.fft.fft(signal2)
    corr = cp.fft.ifft(f1 * cp.conj(f2))
    corr = cp.abs(corr)
    # Normalize the cross-correlation values and return the maximum value as the result
    return cp.max(corr) / (cp.linalg.norm(signal1) * cp.linalg.norm(signal2))


def compute_pca(signal_set, n_components=40):  # n_components is the number of graph signal nodes. You can set it yourself according to the number of graph signal nodes you want to use.
    """
    Use randomized PCA to reduce the dimension of the signal set, maintaining the cumulative variance contribution rate of 90%
    """
    signal_set_np = cp.asnumpy(signal_set)
    pca = PCA(n_components=n_components, svd_solver='randomized')
    pca_transformed = pca.fit_transform(signal_set_np)
    return cp.asarray(pca_transformed)


def compute_pcc_matrix(signals):
    """
    Calculate the Pearson correlation coefficient matrix between the signals after dimensionality reduction
    """
    N = signals.shape[0]
    pcc_matrix = cp.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            pcc_matrix[i, j] = cp.corrcoef(signals[i], signals[j])[0, 1]
            pcc_matrix[j, i] = pcc_matrix[i, j]
    return pcc_matrix


def compute_adjacency_matrix(pcc_matrix, fft_corr_matrix):
    """
    Construct an adjacency matrix, using FFT normalized cross-correlation and Pearson correlation coefficient as weights
    """
    adjacency_matrix = cp.abs(pcc_matrix * fft_corr_matrix)
    cp.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


def compute_degree_matrix(adjacency_matrix):
    """
    Calculate degree matrix
    """
    degree_matrix = cp.diag(cp.sum(adjacency_matrix, axis=1))
    return degree_matrix


def compute_normalized_adjacency_matrix(adjacency_matrix, degree_matrix):
    """
    Calculate the regularized adjacency matrix A_hat = D^(-0.5) * A * D^(-0.5)
    """
    degree_inv_sqrt = cp.diag(1.0 / cp.sqrt(cp.diag(degree_matrix)))
    normalized_adjacency_matrix = degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt
    return normalized_adjacency_matrix


def process_data(file_path, n_components=40): #n_components is the number of graph signal nodes. You can set it yourself according to the number of graph signal nodes you want to use.
    """
    Process the data, calculate the adjacency matrix for each sample, and combine the signal and its adjacency matrix
    """
    data = np.load(file_path)
    labels = data[:, 0]
    signals = data[:, 1:].reshape((data.shape[0], 40, 512)) # (data.shape[0], number of nodes, features of signal) # These can be reset by yourself
    new_data = []
    for i in tqdm(range(signals.shape[0])):
        signal_set = cp.asarray(signals[i])

        # Calculate the FFT normalized cross-correlation coefficient matrix
        fft_corr_matrix = cp.zeros((signal_set.shape[0], signal_set.shape[0]))
        for m in range(signal_set.shape[0]):
            for n in range(m + 1, signal_set.shape[0]):
                fft_corr_matrix[m, n] = compute_fft_correlation(signal_set[m], signal_set[n])
                fft_corr_matrix[n, m] = fft_corr_matrix[m, n]
        signal_set_pca = compute_pca(signal_set, n_components=n_components) # PCA dimensionality reduction
        pcc_matrix = compute_pcc_matrix(signal_set_pca) # Calculate the Pearson correlation coefficient matrix
        A = compute_adjacency_matrix(pcc_matrix, fft_corr_matrix) # Build the adjacency matrix
        D = compute_degree_matrix(A)  # Calculate degree matrix
        A_hat = compute_normalized_adjacency_matrix(A, D) # Calculate the regularized adjacency matrix
        combined = np.hstack((labels[i], signals[i].flatten(), cp.asnumpy(A_hat).flatten()))# Combine the label, signal and adjacency matrix into a new sample
        new_data.append(combined)

    return np.array(new_data)


# Processing the training set
train_data_path = 'D:/documents/GNNTEST/dataset/train/30_train.npy'
train_processed = process_data(train_data_path, n_components=40)
np.save('D:/documents/GNNTEST/dataset/train/cc_30_train_processed.npy', train_processed)

#  Processing the test dataset
test_data_path = 'D:/documents/GNNTEST/dataset/test/30_test_015db.npy'
test_processed = process_data(test_data_path, n_components=40)
np.save('D:/documents/GNNTEST/dataset/test/cc_30_test_processed_015db.npy', test_processed)

print("The adjacency matrix is calculated and saved into a new .npy file！")


