YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 33% of 100 cases. 

MISCLASSIFICATION on training: 6% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	65	2	0
0	4	29	1

MISCLASSIFICATION on test file: 11.5735% of 769 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	506	62	0
0	27	174	1

DECISION TREE (size=11)

F10 <= 2.000000 : 0 (44.0/5.0)
F10 > 2.000000 :
|   F1 <= -0.340000 : 0 (15.0/1.3)
|   F1 > -0.340000 :
|   |   F10 <= 4.700000 :
|   |   |   F1 <= 0.150000 : 0 (4.0/1.2)
|   |   |   F1 > 0.150000 :
|   |   |   |   F1 <= 1.740000 : 1 (22.0/3.7)
|   |   |   |   F1 > 1.740000 : 0 (6.0/2.3)
|   |   F10 > 4.700000 : 1 (9.0/1.3)
