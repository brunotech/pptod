import logging, time, os

class Config:
    def __init__(self, data_prefix):
        # data_prefix = r'../data/'
        self.data_prefix = data_prefix
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):
        self.vocab_path_train = f'{self.data_prefix}/multi-woz-processed/vocab'
        self.data_path = f'{self.data_prefix}/multi-woz-processed/'
        self.data_file = 'data_for_damd.json'
        self.dev_list = f'{self.data_prefix}/multi-woz/valListFile.json'
        self.test_list = f'{self.data_prefix}/multi-woz/testListFile.json'

        self.dbs = {
            'attraction': f'{self.data_prefix}/db/attraction_db_processed.json',
            'hospital': f'{self.data_prefix}/db/hospital_db_processed.json',
            'hotel': f'{self.data_prefix}/db/hotel_db_processed.json',
            'police': f'{self.data_prefix}/db/police_db_processed.json',
            'restaurant': f'{self.data_prefix}/db/restaurant_db_processed.json',
            'taxi': f'{self.data_prefix}/db/taxi_db_processed.json',
            'train': f'{self.data_prefix}/db/train_db_processed.json',
        }
        self.domain_file_path = (
            f'{self.data_prefix}/multi-woz-processed/domain_files.json'
        )
        self.slot_value_set_path = f'{self.data_prefix}/db/value_set_processed.json'

        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi

        self.enable_aspn = True
        self.use_pvaspn = False
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False

        self.exp_domains = ['all'] # hotel,train, attraction, restaurant, taxi
        self.max_context_length = 900
        self.vocab_size = 3000

