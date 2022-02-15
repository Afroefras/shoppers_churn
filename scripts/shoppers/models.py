# Clase madre
from .base import BaseClass

# Control de datos
from typing import Dict
from IPython.display import display

# Ingeniería de variables
from numpy import nan
from datetime import datetime, date
from pandas import read_csv, to_datetime

class ShoppersChurn(BaseClass):
    def __init__(self, file_name: str, main_dict: Dict) -> None:
        super().__init__(file_name)
        self.main_dict = main_dict


    def get_files(self, shopper_id_col: str='shopper_id') -> None:
        data = read_csv(self.file_path, low_memory=False)
        end_of_shopper_data = [x for x,y in enumerate(data.columns) if y=='end_of_shoppers_data'][0]
        self.sh = data.iloc[:,:end_of_shopper_data].drop_duplicates(shopper_id_col)
        self.df = data[[shopper_id_col]].join(data.iloc[:,end_of_shopper_data+1:])


    def clean_shopper_data(self, marital_col: str='marital_status', insurance_col: str='insurance', bank_col: str='bank', transport_col: str='transport') -> None:
        df = self.sh.copy()
        df[marital_col] = df[marital_col].map(lambda x: nan if str(x)=='nan' else x.replace(' ',''))

        aux = []
        for x in df[insurance_col]:
            if str(x)=='nan': aux.append(nan)
            else: 
                try: to_append = to_datetime(x, format=r'%d/%m/%y')
                except: 
                    try: to_append = to_datetime(x[:10], format=r'%Y-%m-%d')
                    except: 
                        try: to_append = to_datetime(x[:11], format=r'%d-%b-%Y')
                        except:
                            try: to_append = to_datetime(x[:11], format=r'%d-%b-%y')
                            except: 
                                print(f'Date: "{x}" was not converted successfully')
                                to_append = nan
                finally: aux.append(to_append)
        df[insurance_col] = aux

        df = self.choose_correct(df, bank_col, self.main_dict[bank_col], n=1, cutoff=0.7)

        df[transport_col] = df[transport_col].map(lambda x: nan if str(x)=='nan' else x.split()[0].title())
        aux = df[transport_col].value_counts(1).to_frame()
        self.main_dict[transport_col] = [x for x,y in zip(aux.index, aux[transport_col]) if y>=0.02]
        df = self.choose_correct(df, transport_col, self.main_dict[transport_col], n=1, cutoff=0.7)

        self.sh = df.copy()


    def vars_shopper(self, id_col: str='shopper_id', official_id_col: str='official_id', insurance_col: str='insurance', last_date_col: str='last_date') -> None:
        df = self.sh.set_index(id_col)

        df['birthday'] = to_datetime(df[official_id_col].str[4:10], format=r'%y%m%d')
        df['birthday'] = df['birthday'].map(lambda x: date(x.year-100, x.month, x.day) if x.year>datetime.today().year else x)
        df['age'] = (datetime.today() - df['birthday']).dt.days/365

        df['genre'] = df[official_id_col].str[10:11]

        df[last_date_col] = to_datetime(df[last_date_col])
        df['is_churn'] = ((datetime.today() - df[last_date_col]).dt.days//7 >= 4)*1
        df['days_for_insurance_exp'] = df[[insurance_col,last_date_col]].apply(lambda x: nan if str(x[0])=='nan' else (x[0] - x[-1]).days, axis=1)
        
        df.drop([official_id_col, 'birthday', insurance_col, last_date_col], axis=1, inplace=True)

        self.main_dict['shop_num_cols'] = df.sample(frac=0.1).describe().columns.tolist()
        self.main_dict['shop_cat_cols'] = [x for x in df.columns if x not in self.main_dict['shop_num_cols']]
        self.main_dict['shop_num_cols'].remove('is_churn')

        self.main_dict['shop_bin_dict'] = {}
        for col in self.main_dict['shop_num_cols']:
            df = self.get_bins(df, col, self.main_dict['shop_bin_dict'], replace_col=True)
            df[col] = df[col].cat.add_categories('Unknown')

        df.fillna('Unknown', inplace=True)
        self.sh = df.astype(str).copy()


    def vars_orders(self, id_col: str='shopper_id') -> None:
        df = self.df.copy()
        df['n_orders'] = 1
        agg = df.pivot_table(index=id_col, values='n_orders', aggfunc=sum)
        for col in df.columns[:-1]:
            if col!=id_col: agg = agg.join(self.bin_distrib(df, id_col, col, self.main_dict), )
        
        self.main_dict['orders_bin_dict'] = {}
        for col in agg.columns:
            agg = self.get_bins(agg, col, self.main_dict['orders_bin_dict'], replace_col=True)
            agg[col] = agg[col].cat.add_categories('Unknown')

        self.total = self.sh.join(agg.fillna('Unknown'))

    def shoppers_train_model(self, n_vars: int=20, **kwargs) -> None:
        df = self.total.copy()
        X = df[[x for x in df.columns if x not in ['is_churn']]].copy()
        y = df['is_churn'].values
        model, _, coefs = self.train_model(X, y, **kwargs)
        # self.profiles(df, cluster_col='is_churn')
        self.main_dict['shopper_model'] = model
        print(f'\nThese are the {n_vars} most relevant variables:')
        display(coefs.head(n_vars//2).append(coefs.tail(n_vars//2)))
        self.total['predict_proba'] = [x[-1] for x in model.predict_proba(X)]
        self.total.to_csv(self.base_dir.joinpath('shopper_predict.csv'))
        # self.tree_to_code(X, y)