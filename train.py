from sklearn.metrics import log_loss
from xgboost import XGBClassifier, XGBModel, DMatrix



# Train each model using the split data
color_id_model = XGBClassifier(enable_categorical=True)
color_id_model.fit(color_id_train,color_id_train_target)
print("1")
mana_model = XGBClassifier(enable_categorical=True)
mana_model.fit(mana_train,mana_train_target)
print("2")
type_model = XGBClassifier(enable_categorical=True)
type_model.fit(type_train,type_train_target)
print("3")
subtype_model = XGBClassifier(enable_categorical=True)
subtype_model.fit(subtype_train,subtype_train_target)
print("4")
keyword_model = XGBClassifier(enable_categorical=True)
keyword_model.fit(keyword_train,keyword_train_target)
print("5")
power_model = XGBClassifier(enable_categorical=True)
power_model.fit(power_train,power_train_target)
print("6")
toughness_model = XGBClassifier(enable_categorical=True)
toughness_model.fit(toughness_train,toughness_train_target)
print("7")

prob_predictions = power_model.predict_proba(power_test)
print(power_test[0])
print(max(prob_predictions[0]))
loss = log_loss(power_test_target,prob_predictions, labels=power_model.classes_)
print(f"Loss: {loss}")


# Test on a sample that does not contain certain values

power_model.predict()