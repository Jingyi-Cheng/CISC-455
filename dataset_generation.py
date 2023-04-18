def create_data():
    import random
    import pandas as pd

    # Define the number of records you want to generate
    num_records = 1000

    # Define the possible values for each feature
    ages = range(18, 91)
    genders = ['male', 'female']
    races = ['white', 'black', 'yellow']
    blood_pressures = ['Low', 'Normal', 'High']
    bmis = [round(random.uniform(15, 40), 1) for _ in range(num_records)]

    # Create an empty list to store the generated records
    data = []

    # Generate each record
    for i in range(num_records):
        age = random.choice(ages)
        gender = random.choice(genders)
        race = random.choice(races)
        blood_pressure = random.choice(blood_pressures)
        bmi = bmis[i]
        # Set the readmission rate based on the values of blood pressure, BMI, and age
        if blood_pressure == 'High' and age >= 50:
            readmission = random.choice([1, 1, 1, 0, 0])  # Higher probability of readmission
        else:
            readmission = random.choice([0, 0, 0, 0, 1])  # Lower probability of readmission
        data.append([age, gender, race, blood_pressure, bmi, readmission])

    # Randomly select 5% of the records and set their readmission rate to 1
    num_high_risk = int(num_records * 0.05)
    high_risk_indices = random.sample(range(num_records), num_high_risk)
    for i in high_risk_indices:
        data[i][-1] = 1

    # Convert the list of records to a pandas dataframe
    df = pd.DataFrame(data, columns=['age', 'gender', 'race', 'blood_pressure', 'bmi', 'readmission'])

    # Save the dataframe as a CSV file
    df.to_csv('dataset.csv', index=False)

    # Preview the generated dataset
    print(df.head(10))


create_data()
