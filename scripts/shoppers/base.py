# Control de datos
from typing import Dict
from pathlib import Path
from IPython.display import display

# Ingeniería de variables
from re import sub, UNICODE
from numpy import nan, array
from unicodedata import normalize
from difflib import get_close_matches
from pandas import DataFrame, qcut

# Modelos
from sklearn.tree import _tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MaxAbsScaler


class BaseClass:
    def __init__(self, file_name: str) -> None:
        self.base_dir = Path.cwd().parent
        self.data_dir = self.base_dir.joinpath('data')
        self.file_name = file_name
        self.file_path = self.data_dir.joinpath(f'{self.file_name}.csv')
        if not self.file_path.is_file():
            print(f'There should be a file called "{self.file_name}" at:\n{self.data_dir}\n\nAdd it and try again!')

    def clean_text(self, text: str, pattern: str="[^a-zA-Z0-9\s]", lower: bool=False) -> str: 
        '''
        Limpieza de texto
        '''
        # Reemplazar acentos: áàäâã --> a
        clean = normalize('NFD', str(text).replace('\n', ' \n ')).encode('ascii', 'ignore')
        # Omitir caracteres especiales !"#$%&/()=...
        clean = sub(pattern, ' ', clean.decode('utf-8'), flags=UNICODE)
        # Mantener sólo un espacio
        clean = sub(r'\s{2,}', ' ', clean.strip())
        # Minúsculas si el parámetro lo indica
        if lower: clean = clean.lower()
        # Si el registro estaba vacío, indicar nulo
        if clean in ('','nan'): clean = nan
        return clean

    def choose_correct(self, df: DataFrame, col: str, correct_list: list, fill_value: str='Otro', keep_nan: bool=True, replace_col: bool=True, **kwargs) -> DataFrame:
        '''
        Recibe un DataFrame y una lista de posibilidades, especificando la columna a revisar
        elige la opción que más se parezca a alguna de las posibilidades
        '''
        # Aplicar limpieza de texto a la lista de posibilidades
        correct_clean = list(map(lambda x: self.clean_text(x, lower=True), correct_list))+['nan']
        # Hacer un diccionario de posibilidades limpias y las originales recibidas
        correct_dict = dict(zip(correct_clean, correct_list+['nan']))

        # Aplicar la limpieza a la columna especificada
        df[f'{col}_correct'] = df[col].map(lambda x: self.clean_text(x,lower=True))
        # Encontrar las posibilidades más parecidas
        df[f'{col}_correct'] = df[f'{col}_correct'].map(lambda x: get_close_matches(str(x), correct_clean, **kwargs))
        # Si existen parecidas, traer la primera opción que es la más parecida
        df[f'{col}_correct'] = df[f'{col}_correct'].map(lambda x: x[0] if isinstance(x,list) and len(x)>0 else nan)
        # Regresar del texto limpio a la posibilidad original, lo no encontrado se llena con "fill_value"
        df[f'{col}_correct'] = df[f'{col}_correct'].map(correct_dict).fillna(fill_value)
        
        if keep_nan: df[f'{col}_correct'] = df[f'{col}_correct'].map(lambda x: nan if str(x)=='nan' else x)
        if replace_col: df = df.drop(col, axis=1).rename({f'{col}_correct':col}, axis=1)
        return df
        
    def two_char(self, n): 
        '''
        Función para convertir float: 1.0 --> str: '01.00'
        '''
        return str(round(n,2)).zfill(2)

    def get_bins(self, df: DataFrame, col: str, bin_dict: Dict, n_bins: int=5, replace_col: bool=False) -> DataFrame:
        # Encontrar el bin al cual el dato pertenece
        df[f'{col}_range'], bins = qcut(df[col], q=n_bins, retbins=True, duplicates='drop')
        
        # Agregar cotas superiores e inferiores para nuevos datos
        bins = list(bins)
        bins.append(min(bins)-100)
        bins.append(max(bins)+100)
        bin_dict[f'{col}_bins'] = array(bins)

        # Convertirlo a texto: [1.0 - 5.0] --> '01.00 a 05.00'
        df[f'{col}_range'] = df[f'{col}_range'].map(lambda x: nan if str(x)=='nan' else self.two_char(x.left)+' to '+self.two_char(x.right))
        
        # Reemplazar columna si es indicado
        if replace_col: df = df.drop(col, axis=1).rename({f'{col}_range':col}, axis=1)
        return df


    def bin_distrib(self, df: DataFrame, id_col: str, col_to_count: str, bin_dict: Dict) -> DataFrame:
        df = self.get_bins(df, col_to_count, bin_dict=bin_dict, replace_col=True)
        df['n'] = 1
        df = df.pivot_table(index=id_col, columns=col_to_count, values='n', aggfunc=sum)
        aux = df.copy()
        aux['total'] = aux.sum(axis=1)
        for col in df.columns: df[col] /= aux['total']
        df.rename({x:f'{col_to_count}_{x}' for x in df.columns}, axis=1, inplace=True)
        return df


    def train_model(self, X: DataFrame, y: array, encoder=CatBoostEncoder, scaler=RobustScaler, model=LogisticRegression, **kwargs) -> tuple: 
        '''
        Escala y entrena un modelo, devuelve el score, el objeto tipo Pipeline y la relevancia de cada variable
        '''
        # Conjunto de entrenamiento y de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.77, random_state=7, shuffle=True)

        # Define los pasos del flujo
        pipe_obj = Pipeline(steps=[
            ('encoder', encoder()),
            ('scaler', scaler()),
            ('model', model(**kwargs))
        ]).fit(X_train,y_train)

        
        # Entrena y guarda el score en test
        test_score = roc_auc_score(pipe_obj.predict(X_test), y_test)
        # Guarda el score en train, para revisar sobreajuste
        train_score = roc_auc_score(pipe_obj.predict(X_train), y_train)

        # Imprime los scores
        print(f"\nScore: {'{:.2%}'.format(test_score)}\nTraining score: {'{:.2%}'.format(train_score)}")

        # Elige la forma de obtener las variables más representativas
        # Ya sea por Regresión Lineal
        try: most_important_features = pipe_obj[-1].coef_[0]
        except: 
            # O por Árbol de decisión, Bosque Aleatorio, XGBoost
            try: most_important_features = pipe_obj[-1].feature_importances_
            # De otro modo, solamente asignar un vector de 0s a este objeto
            except: most_important_features = [0]*len(X.columns)

        # Las ordena descendentemente
        coef_var = DataFrame(zip(X.columns, most_important_features)).sort_values(1, ascending=False).reset_index(drop=True)

        # Devuelve el objeto para clustering, la lista de scores tanto en train como en test y la relevancia de cada variable para el modelo 
        return pipe_obj, (test_score,train_score), coef_var


    def profiles(self, df: DataFrame, cluster_col: str='cluster', to_show: int=3) -> None: 
        '''
        Recibe el resultado del método anterior para mostrar la diferencia numérica y categórica de cada clúster para todas las variables
        '''
        prof = {}
        # Obtener el tipo de variable para cada columna
        df_coltype = df.dtypes

        # Guardar las variables numéricas
        num_cols = [x for x,y in zip(df_coltype.index, df_coltype) if y!=object or x==cluster_col]
        try: num_cols.remove(cluster_col)
        except: pass
        # Promedio de cada variable numérica según el clúster
        if len(num_cols)>0: prof['numeric'] = df.pivot_table(index=cluster_col, values=num_cols)

        # Obtener las variables categóricas
        cat_cols = [x for x in df.columns if x not in num_cols]

        # Columna auxiliar para contar registros
        df['n'] = 1
        for col in cat_cols: 
            # Cuenta de registros para cada variable categórica según el clúster
            prof[col] = df.pivot_table(index=cluster_col, columns=col, aggfunc={'n': sum})

        # Mostrar cada perfilamiento en un DataFrame con formato condicional
        for x in prof.values():
            x = x.fillna(0)
            # Con grupo en renglones, valores de clase en columna
            by_clust = x.copy()
            # Igual que la anterior, pero mostrando el % del total
            perc = x/x.sum().sum()
            # Al revés, clúster en columnas
            by_var = x.T.copy()

            # Mostrar las tres tablas, para notar distribución por clúster, distrib por valor de clase a través de clúster y el % del total
            for i, to_plot in enumerate(zip([by_clust, by_var, perc], ["{:.1f}","{:.1f}","{:.1%}"], [0,0,None])):
                # Desagrupa el tuple que se enumeró
                summary, to_format, to_axis = to_plot
                # Mostrar el número de gráficas indicado para cada tipo de variable
                if i+1 <= to_show:
                    # Aplicar el formato de número y formato condicional asignado
                    display(summary.style.format(to_format).background_gradient('Blues', axis=to_axis))


    def tree_to_code(self, X: DataFrame, y: array) -> None:
        '''
        Revisar para ajuste
        '''
        pipeline, _ , _ = self.train_model(X, y, encoder=OneHotEncoder, scaler=MaxAbsScaler, model=DecisionTreeClassifier)
        tree = pipeline.named_steps['model']
        tree_ = tree.tree_
        feature_names = X.columns
        feature_name = ["undefined!" if i == _tree.TREE_UNDEFINED else feature_names[i] for i in tree_.feature]

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print(f"{indent}if {name} <= {threshold}:")
                recurse(tree_.children_left[node], depth + 1)
                print(f"{indent}else:  # if {name} > {threshold}")
                recurse(tree_.children_right[node], depth + 1)
            else:
                print(f"{indent}return {tree_.value[node]}".format(indent, tree_.value[node]))
        recurse(0, 1)