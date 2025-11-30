YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 31.5789% of 19 cases. 

MISCLASSIFICATION on training: 0% of 19 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	6	0	0
0	0	13	1

MISCLASSIFICATION on test file: 49.6842% of 475 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	97	233	0
0	3	142	1

DECISION TREE (size=7)

F10 <= 2.000000 :
|   F1 <= 0.330000 : 0 (4.0/1.2)
|   F1 > 0.330000 :
|   |   F1 <= 0.570000 : 1 (2.0/1.0)
|   |   F1 > 0.570000 : 0 (2.0/1.0)
F10 > 2.000000 : 1 (11.0/1.3)
