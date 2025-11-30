YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 47.4138% of 116 cases. 

MISCLASSIFICATION on training: 2.58621% of 116 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	59	2	0
0	1	54	1

MISCLASSIFICATION on test file: 18.5% of 800 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	418	124	0
0	24	234	1

DECISION TREE (size=15)

F1 <= -0.140000 : 0 (35.0/2.6)
F1 > -0.140000 :
|   F10 <= 2.700000 :
|   |   F10 <= 1.900000 : 0 (13.0/1.3)
|   |   F10 > 1.900000 :
|   |   |   F1 <= 1.350000 :
|   |   |   |   F1 <= 0.350000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.350000 : 1 (5.0/1.2)
|   |   |   F1 > 1.350000 : 0 (6.0/1.2)
|   F10 > 2.700000 :
|   |   F10 <= 4.400000 :
|   |   |   F1 <= 2.350000 : 1 (21.0/3.7)
|   |   |   F1 > 2.350000 : 0 (4.0/1.2)
|   |   F10 > 4.400000 : 1 (30.0/1.4)
