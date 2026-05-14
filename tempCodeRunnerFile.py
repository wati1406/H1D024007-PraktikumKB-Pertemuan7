

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Konversi label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # ← simpan ke y

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_