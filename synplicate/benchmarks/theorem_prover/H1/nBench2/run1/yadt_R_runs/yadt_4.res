YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 39.2857% of 84 cases. 

MISCLASSIFICATION on training: 0% of 84 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	51	0	0
0	0	33	1

MISCLASSIFICATION on test file: 18.7097% of 775 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	388	132	0
0	13	242	1

DECISION TREE (size=11)

F1 <= -0.790000 : 0 (30.0/1.4)
F1 > -0.790000 :
|   F10 <= 2.900000 :
|   |   F10 <= 2.100000 : 0 (16.0/1.3)
|   |   F10 > 2.100000 :
|   |   |   F1 <= 0.190000 : 0 (3.0/1.1)
|   |   |   F1 > 0.190000 :
|   |   |   |   F1 <= 1.210000 : 1 (4.0/1.2)
|   |   |   |   F1 > 1.210000 : 0 (2.0/1.0)
|   F10 > 2.900000 : 1 (29.0/1.4)
