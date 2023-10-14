import sys
import os
main_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(main_dir)
sys.path.append(project_dir)

from kmer_ml import kmer_ml
import pickle as pkl
#from kmer_ml import get_datafilter, get_datamatrix, gridresearch_kfold, get_shapvalue
import json
import numpy as np

def main(cutoff,numb_files_select,class_label,chunkdata_path,traits_scoary_path):
   
    filepath  = chunkdata_path
    filtered_df = kmer_ml.get_datafilter(class_label,traits_scoary_path)
    row_list = filtered_df.transpose().columns.values

    mtr, voc_col, voc_row, removed_percent =kmer_ml.get_datamatrix(row_list,
                                                            numb_files_select=numb_files_select,
                                                            datapath=filepath,
                                                            cutoff=cutoff)
    # X = mtr
    y = filtered_df['data_type']
    temp = [key for key in voc_col]
    feature_names = temp.copy()
    for i in range(len(temp)):
        feature_names[voc_col[temp[i]]] = temp[i]

    kfold_dataset = kmer_ml.gridresearch_kfold(mtr, y,feature_names)
    bestfeature_indices = kfold_dataset['bestfeature_indices']
    bestfeature = kfold_dataset['bestfeature']
    bestpara = kfold_dataset['bestpara']

    ind = np.argsort(-np.abs(np.array(bestfeature_indices[-1])))
    bf_indices = bestfeature_indices[-1][ind]
    bf = [bestfeature[-1][i] for i in ind]
    sub_bestfeature_indices = bf_indices[0:10]
    sub_bestfeature_name = bf[0:10]
    best_para = bestpara[-1]
    shape_value_dataset = kmer_ml.get_shapvalue(mtr, y, sub_bestfeature_indices, sub_bestfeature_name, best_para, class_label)

   
    pre = ''
    for (k,v ) in class_label.items():
        if v <= 1:
            pre = pre + k
    filename = f'{pre}result_kmer_cutoff{cutoff}_file{numb_files_select}.pkl'

    dataset = {
        'shape_value_dataset': shape_value_dataset,
        'kfold_dataset': kfold_dataset,
        'removed_percent': removed_percent,
        'class_label': class_label,
    }

    with open(filename, 'wb') as f:
        pkl.dump(dataset, f)

if __name__ == '__main__':

    try:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        cutoff = data['cutoff']
        numb_files_select = data['numb_files_select']
        traits_scoary_path = data['traits_scoary_path']
        class_label = data['class_label']
        chunkdata_path = data['chunkdata_path']
    except OSError:
        raise  OSError("Pls input the parameters json file.")
    main(cutoff,numb_files_select,class_label,chunkdata_path,traits_scoary_path)

