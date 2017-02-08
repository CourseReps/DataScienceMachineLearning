# Parking Lot Competition

For this challenge, you will first have to read the data from the provided SQLite database, parkinglot_comp.db

Once you have loaded the training data, you will notice that there are 6 columns: Latitude, Longitude, NUC1, NUC2, NUC3, NUC4.
The Latitude and Longitude columns refer to the actual latitude and longitude of the wireless device from which a packet was captured.
The four NUCx columns are the RSSI values from the four monitoring antennas, arranged as the corners of a rectangle.
The latitude and longitude values for these NUCs are as follows:

| Monitor | Latitude | Longitude |
|------|---|---|
| NUC1 | x | x |
| NUC2 | x | x |
| NUC3 | x | x |
| NUC4 | x | x |

With these, you can easily calculate the distance of the wireless device from the four monitors with the following formula:

distance = 

Next, you have to select a model and train it to predict the distance of a wireless device from a monitoring station based on its RSSI value.
Essentially, for each row in the training table, you will get 4 datapoints corresponding to the RSSI values at the 4 NUCs, respectively paired with the calulated distance from the wireless device to each NUC.

As mentioned in the accompanying Jupyter Notebook, there are three tables in the database: training, cross_validation, and test.
The first two contain all 6 columns, and the test table has the Latitude and Longitude columns removed.

You must train your model using the training table, evaluate your model's accuracy with the cross_validation table, and then create a submission file based on the test table.

The format for submission is in a CSV file with 4 columns: NUC1, NUC2, NUC3, NUC4.
This is to avoid any confusion over indexing
Database and CSV I/O is covered in the Jupyter Notebook.
