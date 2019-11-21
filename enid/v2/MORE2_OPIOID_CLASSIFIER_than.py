import pickle
from sklearn.model_selection import train_test_split

import enid

################################  TRAINING  ###################################

with open('./[MORE2]THAN_OPIOID_4030','rb') as f:
    T = pickle.load(f)
    X = pickle.load(f)
    y = pickle.load(f)
T, T_test, X, X_test, y, y_test = train_test_split(
    T, X, y, test_size=0.05, random_state=42, shuffle=True
)

model = enid.than_clf_lstm.build_model(
    2, 40, 30, hidden_size=128, dropout_prob=0.1, learning_rate=0.0001
)
model = enid.than_clf_lstm.train_model(
    model, T, X, y, batch_size=64, num_epochs=10, model_path="model_lstm",
    dev_sample_percentage=0.01, evaluate_every=100
)

#################################  TESTING  ###################################

y_probs = enid.than_clf_lstm.deploy_model(model, t_test=T_test, x_test=X_test)
enid.visualizations.plot_performance(y_test[:, 0], y_probs)
