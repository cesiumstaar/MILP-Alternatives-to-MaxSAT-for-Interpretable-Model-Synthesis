YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 32.3529% of 68 cases. 

MISCLASSIFICATION on training: 2.94118% of 68 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	45	1	0
0	1	21	1

MISCLASSIFICATION on test file: 18.6567% of 670 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	393	82	0
0	43	152	1

DECISION TREE (size=11)

F10 <= 1.800000 : 0 (25.0/1.3)
F10 > 1.800000 :
|   F1 <= 0.100000 : 0 (13.0/1.3)
|   F1 > 0.100000 :
|   |   F1 <= 1.780000 :
|   |   |   F10 <= 2.400000 :
|   |   |   |   F1 <= 0.350000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.350000 : 1 (6.0/2.3)
|   |   |   F10 > 2.400000 : 1 (16.0/1.3)
|   |   F1 > 1.780000 : 0 (6.0/2.3)
