import pandas as pd
import numpy as np # Es buena práctica usar np como alias estándar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
# --- Nuevas importaciones ---
from sklearn.tree import export_text, plot_tree # Para imprimir/dibujar el árbol
from sklearn.metrics import classification_report, roc_auc_score # Métricas detalladas y AUC
from sklearn.preprocessing import LabelBinarizer # Para calcular AUC multiclase
import matplotlib.pyplot as plt # Para dibujar el árbol

# --- Configuración (igual que antes) ---
#data_dir="./datasets/CIC/"
data_dir="../"
#raw_data_filename = data_dir + "Obfuscated-MalMem2022_labeled.csv"
raw_data_filename = data_dir + "Obfuscated-MalMem2022_labeled.10_percent.csv"

print ("Loading raw data")
# Añadimos 'try-except' por si el archivo no existe
try:
    raw_data = pd.read_csv(raw_data_filename, header=None)
except FileNotFoundError:
    print(f"Error: El archivo '{raw_data_filename}' no se encontró.")
    print("Asegúrate de que la variable 'data_dir' esté configurada correctamente.")
    exit() # Salir si no se puede cargar el archivo

print ("Transforming data")
# Las líneas de factorize siguen comentadas, asumimos datos numéricos

# --- Extracción de características y etiquetas (igual que antes) ---
# Columna 0 tiene nombre específico, se ignora.
# Última columna tiene la clase (Benign, Spyware, Ransomware, Trojan)
# Penúltima columna tiene el tipo (asumimos que también se ignora según el código original)
labels_col_index = raw_data.shape[1] - 1
features_last_col_exclusive = raw_data.shape[1] - 2 # Índice de la primera columna a excluir por la derecha

labels_series = raw_data.iloc[:, labels_col_index]
features = raw_data.iloc[:, 1:features_last_col_exclusive] # Features desde col 1 hasta la antepenúltima

# --- Obtener nombres de clases y características ---
# Los nombres de las clases únicas se extraen de la serie de etiquetas
class_names = labels_series.unique()
print("Unique labels:", class_names, "\n")

# Convertir a numpy array (como antes)
labels = labels_series.values.ravel()

# Obtener nombres de características (usaremos los índices de columna como nombres)
# Los nombres corresponderán a las columnas originales del 1 a features_last_col_exclusive-1
# feature_names = [str(i) for i in range(1, features_last_col_exclusive)]
# O si prefieres usar los nombres de columna del DataFrame 'features' después de crearlo:
df_features = pd.DataFrame(features) # Creamos el DataFrame para X
feature_names = [str(col) for col in df_features.columns] # Nombres basados en el DataFrame

# --- División Train/Test (igual que antes) ---
X_train, X_test, y_train, y_test = train_test_split(df_features, labels, train_size=0.8, test_size=0.2, random_state=42) # Añadido random_state para reproducibilidad
print ("X_train, y_train:", X_train.shape, y_train.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)


# Training, choose model by commenting/uncommenting clf=
print ("Training model")
#clf= RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=102)#, max_features=0.8, min_samples_leaf=3, n_estimators=500, min_samples_split=3, random_state=10, verbose=1)
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None)

trained_model= clf.fit(X_train, y_train)

print ("Score (Train Accuracy): ", trained_model.score(X_train, y_train))

# Predicting
print ("Predicting")
y_pred = clf.predict(X_test)

print ("Computing performance metrics")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

print ("Confusion matrix:\n", results)
print ("Error: ", error)

# NUEVO: Calcular precision, recall y F1
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average='micro')
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average='micro')
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='micro')
print('precision_score: ', precision)
print('recall_score:', recall)
print('f1_score: ', f1)
# --- PASO 5: Imprimir/Visualizar el Árbol de Decisión ---
print("\n" + "="*30)
print("Decision Tree Structure")
print("="*30)

# 5.1: Representación textual del árbol
print("\nText Representation:")
try:
    # Usamos los nombres de características y clases obtenidos antes
    # Asegúrate de que class_names sean strings si export_text los requiere así
    tree_rules = export_text(trained_model, feature_names=feature_names)
    print(tree_rules)
    # Puedes guardar esto en un archivo si es muy largo:
    # with open("decision_tree_rules.txt", "w") as f:
    #     f.write(tree_rules)
except Exception as e:
    print(f"Could not generate text representation: {e}")

# 5.2: Visualización gráfica del árbol (opcional, requiere matplotlib)
print("\nGenerating Graphical Representation (Plot)...")
try:
    plt.figure(figsize=(20, 10)) # Ajusta el tamaño según necesidad
    plot_tree(trained_model,
              feature_names=feature_names,
              class_names=list(map(str, clf.classes_)), # Usa las clases aprendidas por el modelo
              filled=True,        # Colorea los nodos
              rounded=True,       # Nodos redondeados
              fontsize=8)         # Tamaño de fuente
    plt.title("Decision Tree Visualization")
    # Puedes guardar la imagen en un archivo:
    # plt.savefig("decision_tree.png", dpi=300)
    plt.show() # Muestra la gráfica
except Exception as e:
    print(f"Could not generate plot: {e}. Is matplotlib installed?")
# --- PASO 6: Calcular y Mostrar Métricas Mejoradas ---
print("\n" + "="*30)
print("Performance Metrics")
print("="*30)

# 6.1: Matriz de Confusión y Error (como antes)
print ("Computing basic performance metrics")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

print ("\nConfusion matrix:\n", results)
print ("Zero-One Loss (Error Rate): {:.4f}".format(error))
print ("Accuracy: {:.4f}".format(1 - error)) # Accuracy es 1 - Error

# 6.2: Informe de Clasificación Detallado
# Muestra Precision, Recall, F1-Score por clase y promedios (macro, weighted)
print("\nDetailed Classification Report:")
# Usamos las clases aprendidas por el clasificador para las etiquetas del informe
report = classification_report(y_test, y_pred, target_names=list(map(str, clf.classes_)))
print(report)

# Las métricas 'micro' que calculabas antes están incluidas en el accuracy global
# y a veces en los promedios del classification_report (depende de la versión y contexto).
# Si aún las quieres explícitamente:
from sklearn.metrics import precision_score, recall_score, f1_score
precision_micro = precision_score(y_test, y_pred, average='micro')
recall_micro = recall_score(y_test, y_pred, average='micro')
f1_micro = f1_score(y_test, y_pred, average='micro')
print(f"\nMicro Average Precision: {precision_micro:.4f}")
print(f"Micro Average Recall (Accuracy): {recall_micro:.4f}")
print(f"Micro Average F1-Score: {f1_micro:.4f}")

# 6.3: Cálculo del AUC (Área Bajo la Curva ROC)
# AUC se define bien para binario. Para multiclase, usamos estrategias One-vs-Rest (OvR) o One-vs-One (OvO).
# Necesitamos las probabilidades predichas por el modelo.
print("\nCalculating AUC Score (One-vs-Rest strategy):")
# Verificar si el clasificador puede dar probabilidades
if hasattr(clf, "predict_proba"):
    y_prob = clf.predict_proba(X_test)

    # Necesitamos binarizar las etiquetas verdaderas (y_test) para usar roc_auc_score en multiclase OvR
    lb = LabelBinarizer()
    # Asegurarse de que el binarizador conozca todas las clases posibles (usando y_train o clf.classes_)
    lb.fit(clf.classes_) # Ajustar con las clases que el modelo conoce
    y_test_binarized = lb.transform(y_test)

    # Comprobar si hay suficientes clases para calcular AUC (>1 clase)
    if y_test_binarized.shape[1] > 1 and y_prob.shape[1] > 1:
         # Asegurarse de que las formas coincidan (mismo número de clases)
        if y_test_binarized.shape[1] == y_prob.shape[1]:
            try:
                # Calcular AUC con promedio macro y weighted
                auc_ovr_macro = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr', average='macro')
                auc_ovr_weighted = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr', average='weighted')
                print(f"AUC (OvR, macro average): {auc_ovr_macro:.4f}")
                print(f"AUC (OvR, weighted average): {auc_ovr_weighted:.4f}")
            except ValueError as e:
                 # Puede ocurrir si una clase solo tiene una instancia en y_test, etc.
                print(f"Could not calculate AUC score: {e}")
        else:
            print("Could not calculate AUC: Mismatch in number of classes between true labels and predicted probabilities.")
            print(f"  y_test binarized shape: {y_test_binarized.shape}")
            print(f"  y_prob shape: {y_prob.shape}")
            print(f"  Classes known by LabelBinarizer: {lb.classes_}")

    elif y_test_binarized.shape[1] <= 1:
         print("AUC is not defined for datasets with only one class.")
    else: # y_prob.shape[1] <= 1 (raro, pero posible si el modelo predice solo una clase)
         print("AUC calculation failed: Predicted probabilities only cover one class.")

else:
    print("AUC score cannot be calculated because the classifier does not support predict_proba().")

print("\n" + "="*30)
print("End of Analysis")
print("="*30)
