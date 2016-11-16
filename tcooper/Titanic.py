import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
import csv as csv

csv_file_object = csv.reader(open('train.csv',newline=''))
header = csv_file_object.__next__()

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)
# print(data[0])

number_passengers = np.size(data[0::, 1].astype(np.int))
number_survived = np.sum(data[0::, 1].astype(np.int))
proportion_survivors = number_survived / number_passengers
print("Proportion Survived: ", proportion_survivors)

women_only_stats = data[0::, 4] == "female"
men_only_stats = data[0::, 4] != "female"

women_onboard = data[women_only_stats, 1].astype(np.float)
men_onboard = data[men_only_stats, 1].astype(np.float)

num_women = len(women_onboard)
num_men = len(men_onboard)

# print("women_only_stats: ", women_only_stats)
# print("women_onboard: ", women_onboard)

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)
print("Proportion Women Survived: ", proportion_women_survived)
print("Proportion Men Survived: ", proportion_men_survived, "\n")

plt.bar([0.25, 1.25], [proportion_women_survived, proportion_men_survived], 0.5)
plt.ylabel("Proportion Survived")
plt.xticks([0.5, 1.5], ["Women", "Men"])
plt.title("Proportion of Men and Women who Survived")
plt.xlim([0,2])

test_file = open("test.csv")
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()

# prediction_file = open("genderbasedmodel.csv", 'w')
# prediction_file_object = csv.writer(prediction_file)
#
# prediction_file_object.writerow(["PassengerId", "Survived"])
# for row in test_file_object:
#     if row[3] == 'female':
#         prediction_file_object.writerow([row[0], '1'])  # predict 0
#     else:
#         prediction_file_object.writerow([row[0], '0'])  # predict 0
# test_file.close()
# prediction_file.close()

fare_ceiling = 40
data[data[0::, 9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0
fare_bracket_size = 10
number_of_price_brackets = int(fare_ceiling / fare_bracket_size)
number_of_classes = len(np.unique(data[0::, 2]))

survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in range(number_of_classes):
    for j in range(number_of_price_brackets):
        women_only_stats = data[
                            (data[0::, 4] == "female")
                            & (data[0::, 2].astype(np.float)
                               == i+1)
                            & (data[0:, 9].astype(np.float)
                               >= j*fare_bracket_size)
                            & (data[0:, 9].astype(np.float)
                               < (j+1)*fare_bracket_size)
                            , 1]

        men_only_stats = data[
                            (data[0::, 4] == "male")
                            & (data[0::, 2].astype(np.float)
                               == i+1)
                            & (data[0:, 9].astype(np.float)
                               >= j*fare_bracket_size)
                            & (data[0:, 9].astype(np.float)
                               < (j+1)*fare_bracket_size)
                            , 1]
        if len(women_only_stats) > 0:
            survival_table[0, i, j] = np.mean(women_only_stats.astype(np.float))
        else:
            survival_table[0, i, j] = 0

        if len(men_only_stats) > 0:
            survival_table[1, i, j] = np.mean(men_only_stats.astype(np.float))
        else:
            survival_table[1, i, j] = 0

survival_table[survival_table != survival_table] = 0

print("Survival Table: \n", survival_table)

survival_table[survival_table <  0.5] = 0
survival_table[survival_table >= 0.5] = 1


predictions_file = open("genderclassmodel.csv", "w")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for row in test_file_object:
    for j in range(number_of_price_brackets):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3-int(row[1])
            break

        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets-1
        elif row[8] >= j*fare_bracket_size and row[8] < (j+1)*fare_bracket_size:
            bin_fare = j

    if row[3] == 'female':
        p.writerow([row[0], '%d' % int(survival_table[0, int(row[1])-1, bin_fare])])
    else:
        p.writerow([row[0], '%d' % int(survival_table[1, int(row[1])-1, bin_fare])])

test_file.close()
predictions_file.close()

# plt.show()
