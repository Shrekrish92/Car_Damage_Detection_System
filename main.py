from utils_ensemble import *
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, help='Directory where csv files are stored')
parser.add_argument('--topk', type=int, default = 2, help='Top-k number of classes')
args = parser.parse_args()

# Read the CSV files
df_p1 = pd.read_csv('/Users/shreyakrishna/Desktop/ensemble/EfficientNet_results.csv')
df_p2 = pd.read_csv('/Users/shreyakrishna/Desktop/ensemble/MobileNet_results.csv')
df_p3 = pd.read_csv('/Users/shreyakrishna/Desktop/ensemble/ResNet50_results.csv')
df_labels = pd.read_csv('/Users/shreyakrishna/Desktop/ensemble/labels.csv')
print(df_labels.head())
print("****************************")
print(df_p1.head())
print("****************************")

df_merged = pd.merge(df_p1, df_p2, on='Filename')
df_merged = pd.merge(df_merged, df_p3, on='Filename')
print(df_merged.head())
print("****************************")

filenames = df_merged['Filename'].tolist()
p1 = df_merged[['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']].values
p2 = df_merged[['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']].values
p3 = df_merged[['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']].values

ground_truth_labels = df_labels['label']
labels = ground_truth_labels.map({'crack': 0, 'dent': 1, 'glass shatter': 2, 'lamp broken': 3, 'scratch': 4, 'tire flat': 5}).values
'''
root = args.data_directory
if not root[-1]=='/':
    root=root+'/'

p1,labels = getfile(root+"alexnet")
p2,_ = getfile(root+"googlenet")
p3,_ = getfile(root+"resnet")
 # p3,_ = getfile(root+"inception")'''

#Check utils_ensemble.py to see the "labels" distribution. Change according to the dataset used. By default it has been set for the SARS-COV-2 dataset.

#Calculate Gompertz Function Ensemble
top = args.topk #top 'k' classes
predictions = Gompertz(top, p1, p2, p3)
#print(set(labels))
#print(set(predictions))

correct = np.where(predictions == labels)[0].shape[0]
total = labels.shape[0]

# print("Accuracy = ",correct/total)
classes = ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']

'''for i in range(1,7):
    classes.append(str(i))'''
print(classes)

metrics(labels,predictions,classes)

plot_roc(labels,predictions)
