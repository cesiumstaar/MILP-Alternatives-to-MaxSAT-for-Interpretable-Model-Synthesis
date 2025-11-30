# id, age, monthly_income, zip, credit_score, years_of_employment, dependents, approved
import random
import math

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def create_age_income_bias(size):
    training_set = open("loan_acquisition_age_income_bias.csv","w")
     # signature
    training_set.write("\"id\",\"age\",\"monthly_income\",\"zip\",\"credit_score\",\"years_of_employment\",\"dependents\",\"approved\"\n")

    for i in range(size):
        # id = i
        training_set.write(f"{i},")

        # age: if type age 
        age = int(random.normalvariate(30,10))
        while age<18 or age>45:
            age = int(random.normalvariate(30,10))
        training_set.write(f"{age},")

        # monthly income
        monthly_income = truncate(abs(random.normalvariate(6000,3000)),2)
        while monthly_income<1000 or monthly_income>20000:
            monthly_income = truncate(abs(random.normalvariate(6000,3000)),2)
        training_set.write(f"{monthly_income},")

        # zip
        zipcode = random.randint(90001,96162)
        training_set.write(f"{zipcode},")

        # credit score
        credit_score = random.randint(300,850)
        training_set.write(f"{credit_score},")

        # years of employment 
        years_of_empl = random.randint(1,45)
        training_set.write(f"{years_of_empl},")

        # dependents
        dependents = random.randint(1,5)
        training_set.write(f"{dependents},")

        # approval
        if age>=18 and age<30:
            training_set.write(f"0.0")
        elif age>=30 and age<49:
            if monthly_income<6000.0:
                training_set.write(f"0.0")
            else:
                training_set.write(f"1.0")
        else: 
            training_set.write(f"1.0")

        training_set.write("\n")

    training_set.close()


def create_aga_bias(size):
    training_set = open("loan_acquisition.csv","w")

    # signature
    training_set.write("\"id\",\"age\",\"monthly_income\",\"zip\",\"credit_score\",\"years_of_employment\",\"dependents\",\"approved\"\n")

    for i in range(size):
        # id = i
        training_set.write(f"{i},")

        # age: if type age 
        age = int(random.normalvariate(30,10))
        while age<18 or age>45:
            age = int(random.normalvariate(30,10))
        training_set.write(f"{age},")

        # monthly income
        monthly_income = truncate(abs(random.normalvariate(10000,4000)),2)
        while monthly_income<1000 or monthly_income>20000:
            monthly_income = truncate(abs(random.normalvariate(10000,5000)),2)
        training_set.write(f"{monthly_income},")

        # zip
        zipcode = random.randint(90001,96162)
        training_set.write(f"{zipcode},")

        # credit score
        credit_score = random.randint(300,850)
        training_set.write(f"{credit_score},")

        # years of employment 
        years_of_empl = random.randint(1,45)
        training_set.write(f"{years_of_empl},")

        # dependents
        dependents = random.randint(1,5)
        training_set.write(f"{dependents},")

        # approval
        if age>=18 and age<30:
            training_set.write(f"0.0")
        else:
            training_set.write(f"1.0")

        training_set.write("\n")

    training_set.close()


create_age_income_bias(100000)