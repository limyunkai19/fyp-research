import os
import numpy as np
from sklearn.cluster import KMeans

model = ["resnet18", "resnet50", "resnet152", "densenet121", "densenet201", "densenet161"]
pre = [0, 2, 4]
name = []
result = []
for p in pre:
    for m in model:
        name.append("{}_({})".format(m, p))
        result.append("explainability{}explainability_iris_{}_{}".format(os.sep, m, p))

gradcam = []

for res in result:
    gradcam.append(np.load(os.sep.join(["results", res, "gradcam.npy"])))
gradcam = np.array(gradcam)

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(gradcam)

for idx, (n, label) in enumerate(zip(name, kmeans.labels_)):
    print(n, ": ", label)

# for i in range(len(name)):
#     for j in range(len(name)):
#         print("dist: {} - {} = {}".format(name[i], name[j], np.linalg.norm(gradcam[i]-gradcam[j])))
