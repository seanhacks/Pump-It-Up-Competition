import time
import csv
from part1 import run

# Run all possible combinations of num, categorical, and model combinations.

if __name__ == "__main__":
    num = ["StandardScaler", "None"]
    cat = ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]
    models = ["LogisticRegression", "MLPClassifier", "RandomForestClassifier", "HistGradientBoostingClassifier", "GradientBoostingClassifier"]
    means = []

    for m in models:
        for c in cat: 
            for n in num:
                start_time = time.time()
                mean = run(c, n, m)
                end_time = time.time()
                run_time = end_time - start_time 
                means.append((m,c,n,mean, run_time))
    print(means)

    with open('results.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['model','categorical_preprocessing', 'numerical_preprocessing', 'mean_accuracy', 'run_time'])
        for row in means:
            csv_out.writerow(row)

