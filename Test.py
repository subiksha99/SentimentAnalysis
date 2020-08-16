import Clean_Train
from Clean_Train import NB_classifier
from Clean_Train import Y_test, X_test
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

Y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix (Y_test, Y_predict_test)

sns.heatmap(cm, annot = True)
plt.show()
print(classification_report(Y_test, Y_predict_test))
