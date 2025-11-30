YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 33.3333% of 51 cases. 

MISCLASSIFICATION on training: 3.92157% of 51 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	33	1	0
0	1	16	1

MISCLASSIFICATION on test file: 22.205% of 644 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	353	106	0
0	37	148	1

DECISION TREE (size=9)

F1 <= 0.130000 : 0 (24.0/2.5)
F1 > 0.130000 :
|   F10 <= 2.000000 :
|   |   F1 <= 0.570000 :
|   |   |   F1 <= 0.330000 : 0 (2.0/1.0)
|   |   |   F1 > 0.330000 : 1 (2.0/1.0)
|   |   F1 > 0.570000 : 0 (8.0/1.3)
|   F10 > 2.000000 : 1 (15.0/2.5)
