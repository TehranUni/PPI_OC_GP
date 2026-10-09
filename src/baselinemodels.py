
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import  StandardScaler

path = '/content/drive/My Drive/data/'



def ImportTrainData(file_name, year):
  # file_name  = 'N18'
    train_data = pd.read_csv(path+f'main_data/train_data/{year}/train/{file_name}.csv')
    test_data = pd.read_csv(path+f'main_data/train_data/{year}/test/{file_name}.csv')

    return train_data, test_data

def read_data(dataset, year, node_data):
  if year:
    disease_train_data, disease_test_data = ImportTrainData(dataset, year)
  else:
    node_data = pd.read_csv(path+f'main_data/full/ppi_full_emb.csv', sep=",")
    disease_data = ImportTrainData(dataset, '')




  node_data['test'] = 1
  node_data['label'] = 0
  node_data['score'] = 0

  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'label'] = 1
  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'test'] = 0

  node_data.loc[node_data['ensembl'].isin(disease_test_data['ensembl']),'label'] = 1

  positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 0)]
  test_data = node_data.loc[node_data["test"]==1]



  return positive_data, test_data


def compute_auc(df, scores):

  df_copy = df.copy()
  df_copy['score'] = scores
  # df['score'] = scores
  df_sorted = df_copy.sort_values(by='score', ascending=False)
  df_sorted.reset_index(inplace=True, drop=True)
  positive_samples = df_sorted.loc[df_sorted['label'] == 1]

  avg_rank = pd.DataFrame(positive_samples.index).mean()
  AUC = 1 - (avg_rank / df_sorted.shape[0])


  return AUC[0]

def compute_topk_metrics(df: pd.DataFrame, scores, percent: float, label_col='label', score_col='score') -> dict:
    """
    Compute recall, precision, and F1-score for the top-k percent of ranked samples.

    Parameters:
    - df: DataFrame with 'label' and 'score' columns.
    - percent: top-k percentage (e.g., 5, 10, 30).
    - label_col: name of the column containing true labels.
    - score_col: name of the column containing predicted scores.

    Returns:
    - dict with recall, precision, f1, tp, fp, fn, total_positives.
    """

    assert 0 < percent <= 100, "percent must be in (0, 100]"

    df_copy = df.copy()
    df_copy['score'] = scores
    df_sorted = df_copy.sort_values(score_col, ascending=False).reset_index(drop=True)
    if percent == 100:
        top_k_count = percent
    else:
        top_k_count = max(1, int(round((percent / 100.0) * len(df_sorted))))

    top_k = df_sorted.iloc[:top_k_count]

    tp = (top_k[label_col] == 1).sum()
    fp = (top_k[label_col] == 0).sum()
    fn = (df_sorted[label_col] == 1).sum() - tp

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        'top_k_percent': percent,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_positives': (df_sorted[label_col] == 1).sum()
    }


def get_feature_columns(df):
    return [
        col for col in df.columns
        if col not in ['ensembl', 'label', 'score', 'test']
    ]

def train_model(train_data, test_data, kernel, nu , gamma ):
  print(f'{kernel}, {nu} , {gamma}')

  feature_cols = get_feature_columns(train_data)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(
        train_data[feature_cols]
    )
  X_test = scaler.transform(
        test_data[feature_cols]
    )
  model = OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma
    )


  model.fit(X_train)

  scores = model.decision_function(X_test)


  return scores



def main(disease, year, node_data, kernel, nu, gamma):


  train_data, test_data = read_data(disease, year, node_data)

  scores = train_model(train_data, test_data, kernel, nu, gamma)
  scores.to_csv(path+f"models/OCSVM/ppi_full/scores/disease_{year}.csv", index=False)

  AUC = compute_auc(test_data, scores)


  print('-----------------results-------------------------------')


  print(f'AUC = {AUC:.4f}')


  metrics5 = compute_topk_metrics(test_data, scores, percent=5)
  metrics10 = compute_topk_metrics(test_data, scores, percent=10)
  metrics30 = compute_topk_metrics(test_data, scores, percent=30)
  metrics100 = compute_topk_metrics(test_data, scores, percent=100)

  return train_data.shape[0], len(test_data[test_data['label']==1]), len(test_data[test_data['label']==0]), AUC, metrics5['recall'], metrics10['recall'], metrics30['recall'], metrics100['recall']

"""##run"""
feature = 'ppi'
year = 2017
file_name = pd.read_csv(path+f"/main_data/train_data/{year}/disease_summary.csv")
ppi = pd.read_csv(path+f'/main_data/{year}/ppi_{year}_700_emb.csv', sep=",")
node_data = ppi
node_data = node_data.rename(columns={'string_id': 'ensembl'})


results = []

kernel = 'rbf'
nu = 0.05
gamma = 'scale'

for row in file_name['disease_id']:


  print(f'Processing dataset: {row}')
  train_data_size, pos_test_data_size, neg_test_data_size, AUC, r5, r10, r30, r100 = main(row, year, node_data, kernel, nu, gamma)
  results.append({
            'Dataset': row,
            'Train Size': train_data_size,
            'Test Size': pos_test_data_size + neg_test_data_size,
            'Positive Test': pos_test_data_size,
            'Negative Test': neg_test_data_size,
            'AUC': AUC,
            'R@5' : r5,
            'R@10': r10,
            'R@30': r30,
            'R@100' : r100

        })


results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(path+f"/results/OCSVM/OC_{year}_{feature}_CV.csv", index=False)

