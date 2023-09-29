#!/usr/bin/env python
# coding: utf-8

# Author: Ge Zhou
# Email: ge.zhou@kaust.edu.sa

# Function to retrieve all file names in a specified directory
# that start with the string 'chunk' followed by a numerical value.
def file_name_listdir_local(file_dir):
    files_local = []  # Initialize an empty list to hold the file names.
    for files in os.listdir(file_dir):  # Loop through all files in the specified directory.
    #    print(files)  # (Optional) print the name of each file (line commented out).
        # Check if the file name starts with 'chunk' and the rest of the name is a numerical value.
        if files.startswith('chunk') and files[6:].isdigit():
            files_local.append(files)  # If condition met, add the file name to the list.
    return files_local  # Return the list of file names meeting the specified criteria.


# ## Generating a Sparse Matrix and Establishing Cut-off Values (0.01, 0.05, 0.1...) for Filtering
#

# In[ ]:


# Define a function named 'get_datamatrix' with parameters 'row_list', 'files_select', 'datapath', and 'cutoff'
def get_datamatrix(row_list,files_select,datapath='./',cutoff=0.05):
    if not datapath.endswith('./'):
        datapath = datapath +'/'

    # Call file_name_listdir_local function to obtain a list of files from the specified directory
    files_local = file_name_listdir_local(datapath)

    # Restrict the list of files to the first 'files_select' number of files
    if files_select < len(files_local):
        files_local = files_local[0:files_select]

    # Initialize an empty dictionary to store the column vocabulary
    vocabulary_col = {}

    # Initialize an empty dictionary to store the row vocabulary
    vocabulary_row = {}

    # Populate the row vocabulary with the values from 'row_list'
    for rl in row_list:
        _ = vocabulary_row.setdefault(rl, len(vocabulary_row))

    # Initialize counters and empty lists for further processing
    num_removed = 0
    num_preserved  = 0
    row =[]
    col = []
    data = []
    num_list = []  # This list is initialized but not used within the function

    # Iterate through each file in the restricted list of files
    for filename in files_local:
        # Open and read the file in binary mode
        with open(datapath + filename, 'rb') as f:
            ob = f.readlines()

            # Iterate through each line in the file
            for lines in ob:
                n = 0
                linestr = lines.decode('utf-8').split()  # Decode the binary line to utf-8 and split it into words

                # Count the occurrences of words starting with 'assembly' and ending with a digit
                for s in linestr:
                    if s.startswith('assembly'):
                        n += int(s[-1])

                # Check if the ratio of the count to 857.0 is greater than the cutoff value
                if n/857.0 > cutoff:
                    num_preserved += 1  # Increment the count of preserved lines
                    indx_col = vocabulary_col.setdefault(linestr[0], len(vocabulary_col))  # Update the column vocabulary

                    # Process each word in the line
                    for s in linestr:
                        if s.startswith('assembly'):
                            sp = s.split(':')
                            if sp[0][9:] in vocabulary_row:
                                col.append(indx_col)  # Append the column index
                                indx_row = vocabulary_row.get(sp[0][9:])  # Get the row index from the vocabulary
                                row.append(indx_row)  # Append the row index
                                data.append(int(sp[1]))  # Append the data value
                else:
                    num_removed += 1  # Increment the count of removed lines

    # Calculate the percentage of removed lines
    remove_percent = num_removed/(num_preserved + num_removed)

    # Create a Compressed Sparse Row matrix from the data, row, and col lists
    mtr = csr_matrix((data, (row, col)))

    # Return the sparse matrix, column vocabulary, row vocabulary, and removal percentage as output
    return mtr, vocabulary_col, vocabulary_row, remove_percent


# ## Get and Filter Data: Acquire the specified labels from the dataset and filter out the two designated data types.
#    ### HH&HA HH&AA AND HA&AA

# In[12]:


def get_datafilter(datalabel,filepath):
    # Reading data from a CSV file and storing it in a DataFrame called df
    if filepath.endswith('/'):
        df = pd.read_csv(filepath+'shortened_traits_scoary.csv')
    else:
        df = pd.read_csv(filepath + '/shortened_traits_scoary.csv')


    # Extracting values from the input dictionary datalabel and storing them in variables HA, HH, and AA
    HA = datalabel['HA']
    HH = datalabel['HH']
    AA = datalabel['AA']

    # Creating a new column called "data_type" in df.
    # This column is populated based on the values in the "mixed_other", "human_other", and "animal_other" columns of df.
    df["data_type"] = df.apply(lambda row: HA if row["mixed_other"] == 1
                               else (HH if row["human_other"] == 1
                                     else (AA if row["animal_other"] == 1 else None)), axis=1)

    # Filtering df to include only the rows where "data_type" is 0 or 1.
    # Storing the result in a new DataFrame called filtered1_df.
    filtered1_df = df[df['data_type'].isin([0, 1])]

    # Returning the filtered DataFrame.
    return filtered1_df

# ### Testing get_datafilter()

# In[13]:


# # XGBoost model training
# ## Conducting Grid Search and K-Fold Cross Validation with XGBoost

# In[ ]: