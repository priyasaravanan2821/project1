import pickle

model = pickle.load(open('model log_rf_prof.pkl', 'rb'))
# define one new instance
Xnew1 = [[56222,100000,58,1.51,1.324503311,2000,2658,22.43,31.4,125,125,120,10000000,1214.23,20000]]


#Xnew = [[33,	25.27,	0.90,	0,	0,	0,	1,	1,	1,	0,	5,	10,	1,	0,	72,	18,	10.0,	11.8,	0.16,	2.54,	23,	10.52,	49.70,	0.36,	84,120,	80,	13,	15,	18,	20]]
# make a prediction
ynew = model.predict(Xnew1)
print("X=%s, Predicted=%s" % (Xnew1[0], ynew[0]))