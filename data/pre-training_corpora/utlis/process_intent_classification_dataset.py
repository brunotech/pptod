if __name__ == '__main__':
    print ('Processing Intent Classification Datasets...')
    data_list = []
    in_f = r'../raw_data/raw_intent_classification_data/clinc_data.txt'
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        data_list.extend(l.strip('\n') for l in lines)
    in_f = r'../raw_data/raw_intent_classification_data/ATIS_data.txt'
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        data_list.extend(l.strip('\n') for l in lines)
    in_f = r'../raw_data/raw_intent_classification_data/SNIPS_data.txt'
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        data_list.extend(l.strip('\n') for l in lines)
    print (len(data_list))

    import random
    random.shuffle(data_list)

    dev_size = int(0.1 * len(data_list))

    dev_list = data_list[:dev_size]
    train_list = data_list[dev_size:]

    import os
    save_path = r'../separate_datasets/Intent_Classification/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    out_f = f'{save_path}/train_intent_classification_data.txt'
    with open(out_f, 'w', encoding = 'utf8') as o:
        for item in train_list:
            o.writelines(item + '\n')

    out_f = f'{save_path}/test_intent_classification_data.txt'
    with open(out_f, 'w', encoding = 'utf8') as o:
        for item in dev_list:
            o.writelines(item + '\n')
    print ('Processing Intent Classification Datasets Finished!')