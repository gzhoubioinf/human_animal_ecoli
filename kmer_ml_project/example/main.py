import pickle as pkl
from datetime import datetime
from process import get_datafilter,get_datamatrix,gridresearch_kfold,get_shapvalue

if __name__ =='__main__':
    filepath = './dataset'
    # Types with "0" and "1" will be kept for training purposes
    class_label = {'HA':1, 'HH':2, 'AA':0}
    
    filtered_df = get_datafilter(class_label,filepath)
    row_list =  filtered_df.transpose().columns.values
    cutoff = 0.05 # default value
    files_select  = 5 # number of chunk files selected
    mtr, voc_col, voc_row, removed_percent = get_datamatrix(row_list, files_select=files_select,datapath=filepath,cutoff = cutoff)
    # X = mtr
    y = filtered_df['data_type']
    temp = [key for key in voc_col]
    feature_names = temp.copy()
    for i in range(len(temp)):
        feature_names[voc_col[temp[i]]] = temp[i]
    performance,method,report,roc_auc,bestfeature,bestfeature_indices,bestpara = gridresearch_kfold(mtr,y,feature_names)
    
    sub_bestfeature_indices = bestfeature_indices[0][0:10]
    sub_bestfeature_name = bestfeature[0][0:10]
    best_para = bestpara[0]
    shap_values =get_shapvalue(mtr, y,sub_bestfeature_indices,sub_bestfeature_name, best_para, class_label)
    
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime('%Y%m%d_%H%M')
    filename = f'result_kmer_cutoff{cutoff}_file{files_select}_{datetime_string}.pkl'
    
    dataset = {
        'removed_percent':removed_percent,
	    'class_label':class_label,
        'performance': performance,
        'method':method,
        'report':report,
        'roc_auc':roc_auc,
        'bestfeature':bestfeature,
        'bestfeature_indices':bestfeature_indices,
        'bestpara':bestpara
    }
    with open(filename, 'wb') as f:
        pkl.dump(dataset, f)

