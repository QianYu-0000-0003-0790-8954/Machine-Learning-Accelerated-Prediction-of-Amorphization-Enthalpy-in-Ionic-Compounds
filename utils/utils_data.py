# utils_data.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ase import io
from tqdm import tqdm
import itertools

def deload_data(filename, property_names):
    """
    Load and preprocess data for multiple properties.

    Parameters:
    - filename (str): Path to the CSV file containing data.
    - property_names (list): List of property column names to predict.

    Returns:
    - df (pd.DataFrame): DataFrame with structures and properties.
    - species (list): Sorted list of unique species.
    - mean (np.ndarray): Mean values of the properties.
    - std (np.ndarray): Standard deviation of the properties.
    """
    df_ori = pd.read_csv(filename)
    descriptors = pd.read_csv('dataset/features.csv')  # Update path as necessary
    feature_list = [
        'spacegroup_num', 'crystal_system_int', 'density', 'natoms',
        'structural complexity per atom', 'structural complexity per cell',
        'mean absolute deviation in relative bond length', 'max relative bond length',
        'min relative bond length', 'maximum neighbor distance variation',
        'range neighbor distance variation', 'mean neighbor distance variation',
        'avg_dev neighbor distance variation', 'mean absolute deviation in relative cell size',
        'mean Average bond length', 'std_dev Average bond length', 'mean Average bond angle',
        'std_dev Average bond angle', 'mean CN_VoronoiNN', 'std_dev CN_VoronoiNN',
        'vpa', 'packing fraction', 'a', 'b', 'c','natoms','num__ElementProperty_MagpieData maximum NpValence',
    'range oxidation state',
    'Melting Point (K)',
    'num__ElementProperty_MagpieData avg_dev GSbandgap',
    'num__ElementProperty_MagpieData mode NUnfilled',
    'num__ElementProperty_MagpieData mean NUnfilled',
    'num__ElementProperty_MagpieData mean NpUnfilled',
    'num__ElementProperty_MagpieData range SpaceGroupNumber']  # Adjust as needed

    # Create a feature dictionary
    features_dict = {
        str(row['material_id']): [row[fea_name] for fea_name in feature_list if fea_name in row]
        for _, row in descriptors.iterrows()
    }

    props, structures, ids, idxs, features = [], [], [], [], []
    for index, row in tqdm(df_ori.iterrows(), total=df_ori.shape[0], desc="Loading Data"):
        values = row[property_names]
        if not values.isnull().any():
            material_id = str(row['material_id'])
            try:
                struct = io.read(f'dataset/cifs/{material_id}.cif')  # Update path as necessary
                structures.append(struct)
                ids.append(material_id)
                props.append(values.values.astype(float).tolist())
                idxs.append(index)
                features.append(features_dict.get(material_id, [0.0] * len(feature_list)))
            except Exception as e:
                print(f"Error reading {material_id}: {e}")
                continue

    df = pd.DataFrame({
        'structure': structures,
        'prop': props,
        'ids': ids,
        'idx': idxs,
        'feature': features,
        'species': [list(set(s.get_chemical_symbols())) for s in structures]
    })

    if df.empty:
        raise ValueError("No data loaded. Please check your input CSV and file paths.")

    species = sorted(set(itertools.chain.from_iterable(df['species'])))
    props_array = np.array(props)
    mean = props_array.mean(axis=0)
    std = props_array.std(axis=0)
    print('Mean of properties:', mean)
    print('Standard deviation of properties before adjustment:', std)

    # Handle small or zero std values
    std_threshold = 1e-6
    std = np.where(std < std_threshold, 1.0, std)
    print('Adjusted standard deviation of properties:', std)

    return df, species, mean, std

def train_valid_test_split(df, species, valid_size, test_size, seed=12, plot=False):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing all data.
    - species (list): List of unique species.
    - valid_size (float): Fraction of data to use for validation.
    - test_size (float): Fraction of data to use for testing.
    - seed (int): Random seed for reproducibility.
    - plot (bool): Whether to plot the distribution of species in splits.

    Returns:
    - idx_train (list): List of training indices.
    - idx_valid (list): List of validation indices.
    - idx_test (list): List of test indices.
    """
    # Create a stratification label by joining sorted species
    stratify_labels = df['species'].apply(lambda x: ','.join(sorted(x)))

    # Check class distribution
    class_counts = stratify_labels.value_counts()
    insufficient_classes = class_counts[class_counts < 2].index.tolist()
    if insufficient_classes:
        print(f"Warning: {len(insufficient_classes)} classes have fewer than 2 samples. Stratification will be adjusted.")
        # Replace labels of insufficient classes with 'Others'
        stratify_labels = stratify_labels.apply(lambda x: 'Others' if x in insufficient_classes else x)

    try:
        idx_train, idx_valid_test = train_test_split(
            df.index, 
            test_size=valid_size + test_size, 
            random_state=seed, 
            stratify=stratify_labels
        )
    except ValueError as e:
        print(f"Stratification failed: {e}. Proceeding without stratification.")
        idx_train, idx_valid_test = train_test_split(
            df.index, 
            test_size=valid_size + test_size, 
            random_state=seed, 
            stratify=None
        )

    if test_size > 0:
        relative_test_size = test_size / (valid_size + test_size)
        try:
            stratify_valid_test = stratify_labels.loc[idx_valid_test]
            idx_valid, idx_test = train_test_split(
                idx_valid_test, 
                test_size=relative_test_size, 
                random_state=seed, 
                stratify=stratify_valid_test
            )
        except ValueError as e:
            print(f"Stratification for validation/test split failed: {e}. Proceeding without stratification for this split.")
            idx_valid, idx_test = train_test_split(
                idx_valid_test, 
                test_size=relative_test_size, 
                random_state=seed, 
                stratify=None
            )
    else:
        idx_valid, idx_test = idx_valid_test, []

    print(f"Number of training examples: {len(idx_train)}")
    print(f"Number of validation examples: {len(idx_valid)}")
    print(f"Number of testing examples: {len(idx_test)}")
    print(f"Total number of examples: {len(idx_train) + len(idx_valid) + len(idx_test)}")
    assert len(set(idx_train).intersection(idx_valid)) == 0
    assert len(set(idx_train).intersection(idx_test)) == 0
    assert len(set(idx_valid).intersection(idx_test)) == 0

    # Ensure that indices are returned as lists using list() constructor
    idx_train = list(idx_train)
    idx_valid = list(idx_valid)
    idx_test = list(idx_test)

    return idx_train, idx_valid, idx_test