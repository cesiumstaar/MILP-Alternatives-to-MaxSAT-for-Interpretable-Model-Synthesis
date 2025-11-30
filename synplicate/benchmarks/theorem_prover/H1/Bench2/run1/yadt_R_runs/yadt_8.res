YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 43.9189% of 148 cases. 

MISCLASSIFICATION on training: 2.7027% of 148 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	81	2	0
0	2	63	1

MISCLASSIFICATION on test file: 15.9209% of 961 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	518	129	0
0	24	290	1

DECISION TREE (size=17)

F1 <= -0.140000 : 0 (53.0/3.8)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F1 <= 0.920000 :
|   |   |   F1 <= 0.400000 : 0 (4.0/1.2)
|   |   |   F1 > 0.400000 : 1 (9.0/2.4)
|   |   F1 > 0.920000 : 0 (19.0/1.3)
|   F10 > 2.700000 :
|   |   F10 <= 4.600000 :
|   |   |   F1 <= 1.880000 :
|   |   |   |   F1 <= -0.120000 : 0 (2.0/1.0)
|   |   |   |   F1 > -0.120000 : 1 (24.0/1.3)
|   |   |   F1 > 1.880000 :
|   |   |   |   F10 <= 3.700000 : 0 (5.0/1.2)
|   |   |   |   F10 > 3.700000 : 1 (3.0/2.1)
|   |   F10 > 4.600000 : 1 (29.0/1.4)
