# HOW TO SETUP!!!!!!!!!!!
1: Extract the .zip into a pycharm
2: Create a new python environment and run the requirements.txt file using the following commands:
2a: Create python environment: python -m venv fer
2b: Activate environment: .\fer\Scripts\activate
2c: pip install -r (location to file)\requirements.txt
3: Make sure you can your port 5432 is open
4: Run application.py file
5: Go to local browser and enter 127.0.0.1:5000
6: Sign up or login
7: Create a Session
8: Visualize the data
9: View your sessions by clicking your name on the home page
10: Generate PDFs on the Visualize data page (NOTE YOU NEED TO INSTALL WKHTMLTOPDF FOR THIS TO WORK).

If you wish to create your own model you can use the TrainModel file to train one, but this is very cpu/gpu heavy. You can edit the parameters in the CNN Model or train model file to adjust your models accuracy. You could also adjust the number of trials and range in the Hyperparameter file to get better accurate parameters for training a model.