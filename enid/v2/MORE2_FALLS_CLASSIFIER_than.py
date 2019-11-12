import pickle
from sklearn.model_selection import train_test_split

import enid

################################  TRAINING  ###################################

with open('./[MORE2]THAN_FALL_4030','rb') as f:
    T = pickle.load(f)
    X = pickle.load(f)
    y = pickle.load(f)
T, T_test, X, X_test, y, y_test = train_test_split(T, X, y, test_size=0.05, random_state=42, shuffle=True)

model = enid.than_clf.build_model(2, 40, 30, batch_size=64, d_model=256, d_ff=2048, h=8, encoder_layers=1,
                                  hidden_size=128, dropout_prob=0.1)
model = enid.than_clf.train_model(model, T, X, y, learning_rate=0.0001, num_epochs=3, model_path="model", dev_sample_percentage=0.01, evaluate_every=200)

#################################  TESTING  ###################################

y_probs = enid.than_clf.deploy_model(model, t_test=T_test, x_test=X_test)
enid.visualizations.plot_performance(y_test[:, 0], y_probs)
