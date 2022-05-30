import pickle

model = pickle.load(open('model log reg.pkl', 'rb'))
# define one new instance
Xnew1 = [[33,	68.8,	165,	25.27089073,	11,	2,	5,	1,	0,	40,	366,	0.9,	0,	0,	0,	1,	1,	1,	0]]


#Xnew = [[33,	25.27,	0.90,	0,	0,	0,	1,	1,	1,	0,	5,	10,	1,	0,	72,	18,	10.0,	11.8,	0.16,	2.54,	23,	10.52,	49.70,	0.36,	84,120,	80,	13,	15,	18,	20]]
# make a prediction
ynew = model.predict(Xnew1)
print("X=%s, Predicted=%s" % (Xnew1[0], ynew[0]))


