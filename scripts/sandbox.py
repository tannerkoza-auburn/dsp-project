# %%
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from dataclasses import dataclass
from matplotlib.cm import coolwarm
from tqdm import tqdm
from navtools.dsp import parcorr
from scipy.io import loadmat, savemat
from scipy import signal

# Training Data
PERCENTAGE_OF_POPULATION = 75
IS_PLOTTED = True

# Butterworth Filter
FILTER_ORDER = 10
CUTOFF_FREQUENCY = 0.175

fmri_data = loadmat("/home/tannerkoza/devel/dsp-project/data/data.mat")


@dataclass(frozen=True)
class Features:
    intensity: np.ndarray
    filtered_intensity: np.ndarray


def main():
    features = defaultdict()
    
    for population, data in fmri_data.items():

        if type(data) is np.ndarray:
            data = data.flatten()
            percentage_index = int((PERCENTAGE_OF_POPULATION / 100) * data.size)
            features[population] = extract_features(
                population_name=population, data=data[:percentage_index]
            )

    savemat(file_name="/home/tannerkoza/devel/dsp-project/data/features.mat", mdict=features)


def extract_features(population_name: str, data: np.ndarray):
    # initialization
    _, nregions = data[0].shape  # extracted in case # of regions change for datasets

    intensity = np.zeros(nregions)
    filtered_intensity = np.zeros(nregions)

    for person in tqdm(iterable=data, desc=f"processing {population_name} subjects"):
        # process unfiltered data
        intensity = compute_intensity(person=person)
        normalized_person = mean_normalize(person=person)

        # process filtered data
        filtered_person = filter_samples(person=person)
        filtered_intensity = compute_intensity(person=filtered_person)
        normalized_filtered_person = mean_normalize(person=filtered_person)

    return Features(intensity=intensity, filtered_intensity=filtered_intensity)


def filter_samples(person: np.ndarray):
    sos = signal.butter(N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="low", output="sos")
    filtered = signal.sosfiltfilt(sos, person, axis=0)

    return filtered


def compute_intensity(person: np.ndarray):
    intensity = np.sum(np.abs(person) ** 2, axis=0)

    return intensity


def correlate(person: np.ndarray):
    correlation_plane = [parcorr(region, person).sum(axis=1) for region in person]

    return correlation_plane


def mean_normalize(person: np.ndarray):
    max_per_region = np.max(person, axis=0)
    min_per_region = np.min(person, axis=0)
    range_per_region = max_per_region - min_per_region
    mean_per_region = np.mean(person, axis=0)

    normalized_person = (person - mean_per_region) / range_per_region

    return normalized_person

# def plot_features(features: Features):
#     fig, axes = plt.subplots(nrows=1, ncols=2)
    
    
#     for ax, corr in zip(axes.flat, corrs):
    
    
    
    
# regions = np.arange(NREGIONS) + 1
#         X, Y = np.meshgrid(regions, regions)
#         subset_correlation = np.sum(planes, axis=0)
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#         surf = ax.plot_surface(
#             X, Y, subset_correlation, cmap=coolwarm, linewidth=0, antialiased=False
#         )
#         plt.show()

if __name__ == "__main__":
    main()

# %%
