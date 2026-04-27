import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.manifold import TSNE
import seaborn as sns

#caminho
base_path = r"C:\Users\hakaw\OneDrive\Documentos\orl_faces"

#parametros corretos
img_size = (92, 112)

#leitura das imagens
data = []
labels = []

for i in range(1, 41):
    folder = os.path.join(base_path, f"s{i}")
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("L")
        data.append(np.asarray(img).flatten())
        labels.append(i - 1)

data = np.array(data)
labels = np.array(labels)

#PCA
pca = PCA(n_components=50)
data_pca = pca.fit_transform(data)

#SVM com kernel RBF
svm = SVC(kernel='rbf', probability=True)
svm.fit(data_pca, labels)

#selecao de imagem
person_id = np.random.randint(1, 41)
folder = os.path.join(base_path, f"s{person_id}")
files = os.listdir(folder)
file_test = np.random.choice(files)

#imagem de teste
img_path = os.path.join(folder, file_test)
test_img = Image.open(img_path).convert("L")
test_img_array = np.asarray(test_img).flatten()
test_pca = pca.transform([test_img_array])

#previsso
pred_label = svm.predict(test_pca)[0]
pred_probs = svm.predict_proba(test_pca)[0]
top5_idx = np.argsort(pred_probs)[::-1][:5]

#imagem de teste isolada
plt.figure(figsize=(4, 4))
plt.imshow(test_img_array.reshape((112, 92)), cmap='gray')
plt.title(f"Imagem de Teste\nPasta: s{person_id}")
plt.axis('off')
plt.tight_layout()
plt.show()

#imagem de teste + outras 9
plt.figure(figsize=(10, 5))
plt.subplot(2, 5, 1)
plt.imshow(test_img_array.reshape((112, 92)), cmap='gray')
plt.title(f"Teste\nPasta: s{person_id}\nPrevisto: s{pred_label + 1}")
plt.axis('off')

samples_found = 0
for file in files:
    if file != file_test:
        img = Image.open(os.path.join(folder, file)).convert("L")
        img_array = np.asarray(img).flatten()
        samples_found += 1
        plt.subplot(2, 5, samples_found + 1)
        plt.imshow(img_array.reshape((112, 92)), cmap='gray')
        plt.axis('off')
        if samples_found == 9:
            break

plt.tight_layout()
plt.show()

#top 5 classes mais provaveis
print("Top-5 classes mais prováveis:")
for idx in top5_idx:
    print(f"Classe s{idx+1}: {pred_probs[idx]*100:.2f}%")

#matriz de confusao com validacao cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(svm, data_pca, labels, cv=cv)
cm = confusion_matrix(labels, y_pred)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, colorbar=False)
plt.title("Matriz de Confusão (5-fold CV)")
plt.tight_layout()
plt.show()

#t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(data_pca)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_proj[:, 0], y=tsne_proj[:, 1], hue=labels, palette='tab20', legend=None)
plt.title("Projeção t-SNE das Imagens (PCA reduzido)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.tight_layout()
plt.show()
