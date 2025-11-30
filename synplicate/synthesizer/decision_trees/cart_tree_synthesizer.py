from sklearn import tree 
import graphviz
import os



def synthesize(data_file_path, class_names, dir_path, depth,file_name):

    inputs = []
    outputs = []

    # Read input and output from csv
    data_file = open(data_file_path,"r")

    feature_names = []
    first_line = True
    for line in data_file.readlines():
        # Retrieve names
        if first_line:
            names = line.rstrip('\x00\n').split(",")
            feature_names = names[:-1]
            first_line = False
        else:
            line_values = line.rstrip('\x00\n').split(",")
            input_values = list(map(float,line_values[:-1]))
            inputs.append(input_values)
            outputs.append(list(map(int,line_values[-1])))
    

    # print(feature_names)
    # print(inputs)
    # print(outputs)
    data_file.close()

    if depth ==-1:
        dt = tree.DecisionTreeClassifier()
    else:
        dt = tree.DecisionTreeClassifier(max_depth=depth)

    dt = dt.fit(inputs, outputs)

    count =0 
    prediction = dt.predict(inputs)
    for i in range(len(outputs)):
        if outputs[i]!=prediction[i]:
            count +=1

    misclassification_rate = count/len(outputs)
    print(f"Synthesized decision tree with misclassification rate: {misclassification_rate}")

    # Visualization
    os.system(f"mkdir {dir_path}/cart") 
    dot_data = tree.export_graphviz(dt, out_file=None, feature_names=feature_names, class_names=class_names)
    graph = graphviz.Source(dot_data)
    graph.render(f"{dir_path}/cart/cart_dt_{file_name}")

    os.system(f"dot -Tpng {dir_path}/cart/cart_dt_{file_name} -o {dir_path}/cart/cart_dt_{file_name}.png")

    return dt


def evaluate(data_file_path,dt):

    inputs = []
    outputs = []

    # Read input and output from csv
    data_file = open(data_file_path,"r")

    feature_names = []
    first_line = True
    for line in data_file.readlines():
        # Retrieve names
        if first_line:
            names = line.rstrip('\x00\n').split(",")
            feature_names = names[:-1]
            first_line = False
        else:
            line_values = line.rstrip('\x00\n').split(",")
            input_values = list(map(float,line_values[:-1]))
            inputs.append(input_values)
            outputs.append(list(map(int,line_values[-1])))
    

    # print(feature_names)
    # print(inputs)
    # print(outputs)
    data_file.close()


    count =0 
    prediction = dt.predict(inputs)
    for i in range(len(outputs)):
        if outputs[i]!=prediction[i]:
            count +=1

    misclassification_rate = count/len(outputs)

    return misclassification_rate

