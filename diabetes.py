# Özge Çinko

####################
# İş Problemi
####################
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

####################
# Veri Seti Hikayesi
####################
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

###############################
# Görev 1: Keşifçi Veri Analizi
###############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv("datasets/diabetes.csv")
    return data

df = load()
df.head()

# Adım 1: Genel resmi inceleyiniz.
def analyze_data(df):
    """
    Prints detailed information about a pandas DataFrame.

    Parameters:
    df (pandas DataFrame): The DataFrame to analyze.
    """
    print("#" * 30)
    print("Data Shape")
    print("-" * 30)
    print(df.shape)
    print("#" * 30)
    print("Data Columns")
    print("-" * 30)
    print(df.columns)
    print("#" * 30)
    print("Data Types")
    print("-" * 30)
    print(df.dtypes)
    print("#" * 30)
    print("Data Missing Values")
    print("-" * 30)
    print(df.isnull().sum())
    print("#" * 30)
    print("Data Summary Statistics")
    print("-" * 30)
    print(df.describe().T)
    print("#" * 30)


analyze_data(df)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
numeric_variables = [col for col in df.columns if df[col].dtypes != "O"]
categorical_variables = [col for col in df.columns if df[col].dtypes == "O"]
# Outcome int gözüken kategorik bir değişken, bu yakalama şekli geliştirilmelidir.

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
df.groupby("Outcome")[num_cols].agg(["mean", "min", "max"])


# Adım 5: Aykırı gözlem analizi yapınız.
for col in df.columns:
    sns.boxplot(x=df[col])
    plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.columns:
    print(col, check_outlier(df, col))



# Adım 6: Eksik gözlem analizi yapınız.
df.isnull().values.any() # False
df.isnull().sum() # 0


# Adım 7: Korelasyon analizi yapınız.
corr = df[num_cols].corr()
# Korelasyon: Değişkenlerin birbiriyle ilişkisini ifade eden istatistiksel ölçümdür.
# -1, +1 değerlerine yakınsa ilişki o kadar şiddetlidir.
# Birbiriyle yüksek korelasyonu olan değişkenleri genelde beraber çalıştırmamak isteriz.
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, cmap="Greens", ax=ax, square=True)
plt.yticks(rotation=1)
plt.show()
# Korelasyon çıktısı almamıza yarar.


###############################
# Görev 2: Feature Engineering
###############################
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

# Aykırı değerleri baskıladım.
for col in num_cols:
    replace_with_thresholds(df, col)

# Aykırı değerler baskılanmış mı diye kontrol ettim.
for col in df.columns:
    print(col, check_outlier(df, col))

len(df[df["Insulin"] == 0]) # İnsülin 0 olan veri sayısı.
len(df[df["Glucose"] == 0]) # Glikoz 0 olan veri sayısı.

# Öncelikle hangi numerik değişkenlerde kaç adet 0 var bakmak istedim.
for col in num_cols:
    zero_value_length = len(df[df[col]==0])
    print(col, zero_value_length)
    if zero_value_length != 0: # 0 değeri bulunuyorsa,
        df[col] = df[col].replace(0, np.nan) # NaN'a çevirdim.

df.isnull().sum() # Değişkenlerin NaN'a çevrilmesini kontrol ettim.
# Pregnancies 111, SkinThickness 227 ve Insulin 374 adet boş değer oluştu.

# Eksik veri yapısını inceledim.
msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()
#  SkinThickness ve Insulin arasındaki hücre 0.7'dir, iki sütundaki eksik değerler arasında güçlü bir pozitif korelasyon olduğu anlamına gelir.

# Eksik değerleri tahmine dayalı doldurdum.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

# Değişkenlerin standartlaştırılması.
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# KNN'in uygulanması.
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["Age"] = dff["Age"]
df["Insulin"] = dff[["Insulin"]]
df["SkinThickness"] = dff[["SkinThickness"]]
df.head()

# Adım 2: Yeni değişkenler oluşturunuz.
df["LifeStage"] = pd.qcut(df["Age"], 3, labels=["Early Adulthood", "Middle Adulthood", "Late Adulthood"])
df["BMICategory"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, float('inf')], labels=["Underweight", "Normal", "Overweight", "Obese"])

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
df.groupby("BMICategory")["Outcome"].mean()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# One Hot Encoder.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()
df.head()


# Rare Encoder.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Outcome", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "Outcome", cat_cols)

df.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

# Adım 5: Model oluşturunuz.
# X bağımsız değişkenler, Y bağımlı değişken.

y = df["Outcome"]
X = df.drop(["LifeStage", "BMICategory", "Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)