import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
from collections import defaultdict
from tqdm import tqdm

# I/O
INPUT_FILE = "data.mat"
OUTPUT_FILE = "features.mat"

# Data Subsets
IS_SUBSET_PERCENTAGE = True
PERCENTAGE_OF_POPULATION = 100
NPEOPLE = 175

# Principal Component Analysis
N_COMPONENTS = 10  # 10

# Gaussian Blur
SIGMA = 0.001


def compute_raw_signal_features(person: np.ndarray):
    xcorr = signal.correlate(person, person, mode="same")
    xcorr_power = np.sum(np.abs(xcorr) ** 2, axis=0)

    return xcorr_power


def compute_psd_features(person: np.ndarray):
    nsamples, _ = person.shape
    frange, pxx = signal.welch(
        person, nperseg=nsamples, axis=0
    )  # normalized power spectral density

    energy = np.sum(pxx, axis=0)  # region energy
    max_power = np.max(pxx, axis=0)  # region maximum power
    min_power = np.min(pxx, axis=0)  # region minimum power

    findices = np.argmax(pxx, axis=0)
    fmax = frange[findices]  # region frequency with maximum power

    summed_peak_power = []
    for region in pxx.T:
        peak_indices, _ = signal.find_peaks(region)
        summed_peak_power.append(np.sum(region[peak_indices]))

    summed_peak_power = np.array(summed_peak_power)

    return energy, max_power, min_power, fmax, summed_peak_power


def perform_pca(features: np.ndarray):
    pca = PCA(n_components=N_COMPONENTS)
    transform = pca.fit_transform(features)

    return transform


def extract_features(population_name: str, data: np.ndarray):
    # initialization
    raw_features = []

    for person in tqdm(iterable=data, desc=f"processing {population_name} subjects"):
        # feature extraction
        xcorr_power = compute_raw_signal_features(person=person)
        (
            energy,
            max_power,
            min_power,
            fmax,
            summed_peak_power,
        ) = compute_psd_features(person=person)

        # feature logging
        raw_features.append(
            np.concatenate(
                [
                    energy,
                    max_power,
                    min_power,
                    fmax,
                    xcorr_power,
                    summed_peak_power,
                ]
            )
        )

    raw_features = np.array(raw_features)

    # perform principal component analysis
    transformed_features = np.array(perform_pca(features=raw_features))

    # add Gaussian blur
    noise = SIGMA * np.random.randn(*transformed_features.shape)
    output_features = transformed_features + noise

    return output_features


def main():
    # import data
    data_dir_path = pl.Path(__file__).parents[1] / "data"
    raw_signal_data = loadmat(data_dir_path / INPUT_FILE)

    # initialize output
    features = defaultdict()

    for population, data in raw_signal_data.items():
        if type(data) is np.ndarray:
            data = data.flatten()

            if IS_SUBSET_PERCENTAGE:
                npeople = int((PERCENTAGE_OF_POPULATION / 100) * data.size)
            else:
                npeople = NPEOPLE

            people_indices = np.random.choice(data.size, size=npeople, replace=False)
            data = data[people_indices]

            features[population] = extract_features(
                population_name=population, data=data
            )

    # export features
    savemat(file_name=data_dir_path / OUTPUT_FILE, mdict=features)


if __name__ == "__main__":
    main()


# def plot():
#     _, (pxx_ax0, pxx_ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
#         pxx_ax0.semilogy(frange, pxx[:, 0:10])
#         pxx_ax1.semilogy(frange_standardized, pxx_standardized[:, 0:10])

#         _, signal_ax = plt.subplots()
#         signal_ax.plot(person0[:, 0], "r")
#         signal_ax.plot(filtered_person0[:, 0], "g")
#         signal_ax.plot(standardized_person0[:, 0], "b")
#         signal_ax.plot(normalized_person0[:, 0], "m")
#         plt.show()
