YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 1 (2 classes)
MAJORITY MISCLASSIFICATION on training: 45% of 20 cases. 

MISCLASSIFICATION on training: 0% of 20 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	9	0	0
0	0	11	1

MISCLASSIFICATION on test file: 30% of 180 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	63	51	0
0	3	63	1

DECISION TREE (size=7)

F10 <= 3.000000 :
|   F1 <= 0.380000 : 0 (7.0/1.3)
|   F1 > 0.380000 :
|   |   F10 <= 1.700000 : 0 (2.0/1.0)
|   |   F10 > 1.700000 : 1 (4.0/1.2)
F10 > 3.000000 : 1 (7.0/1.3)
