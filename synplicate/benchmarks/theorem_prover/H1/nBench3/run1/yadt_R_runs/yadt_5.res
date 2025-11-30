YaDT version 2.2.0 (c) S. Ruggieri 2002-2019

MAJORITY CLASS value: 0 (2 classes)
MAJORITY MISCLASSIFICATION on training: 34% of 100 cases. 

MISCLASSIFICATION on training: 2% of 100 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	64	2	0
0	0	34	1

MISCLASSIFICATION on test file: 11.399% of 386 cases. 
?	0	1	predicted(cols)/actual(rows)
0	0	0	?
0	235	40	0
0	4	107	1

DECISION TREE (size=15)

F1 <= 0.140000 : 0 (33.0/1.4)
F1 > 0.140000 :
|   F10 <= 2.000000 :
|   |   F10 <= 1.500000 : 0 (19.0/1.3)
|   |   F10 > 1.500000 :
|   |   |   F1 <= 0.840000 :
|   |   |   |   F1 <= 0.400000 : 0 (2.0/1.0)
|   |   |   |   F1 > 0.400000 : 1 (4.0/1.2)
|   |   |   F1 > 0.840000 : 0 (7.0/1.3)
|   F10 > 2.000000 :
|   |   F10 <= 2.700000 :
|   |   |   F1 <= 1.440000 : 1 (11.0/3.6)
|   |   |   F1 > 1.440000 : 0 (3.0/1.1)
|   |   F10 > 2.700000 : 1 (21.0/1.3)
