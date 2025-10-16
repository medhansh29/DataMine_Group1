# DataMine_Group1: How to use this Repository

## File Description:
1. *alerce_client.py*: This is the main client-side interacting file that allows us to fetch lightcurve data for N number of objects through a CLI interface. We will be running this for close to 10k entries once our filters have been set and the code for them has been written down. 
2.  *requirements.txt*: This is a text file that keeps a record of all the libraries we need to have locally to run our code. The advantage of doing this is a pip command called pip install -r requirements.txt

## How to run this repository:

### Downloading Cursor(recommended): 
Cursor is an IDE that gives you access to an integrated AI Agent panel, significantly boosting code efficiency and making it easy to code out sophisticated applications even if you have not coded before.

*Download cursor from here*: https://cursor.com/download

## Installing Python(If Needed):
You also might need to install Python if you have never used an IDE or run Python code locally before

*Download Python from here*: https://www.python.org/downloads/

Then run this command on your IDE terminal:

`python --version` (for Windows)

`python3 --version` (for Mac)


### Cloning the repository from GitHub
Once you have set up an IDE on your local computer, run the following steps in your terminal inside your IDE:

Note: Make sure you are in the correct directory you want to be in. You can check that by looking at the text showing up on an empty terminal. Something like this: PS C:\Users\medha\OneDrive - purdue.edu\Desktop\DataMinePhysics>

1. `git clone https://github.com/{your_github_username}/DataMine_Group1.git`

### Running the project
Once you have cloned the repository into your local IDE we can now run our files.

1. Run `pip install -r requirements.txt`
2. Run the file alerce_client.py as a test to see if it works for you, run it for something like 5-10 objects and see if a CSV is made. The CSV has all the lightcurve data, so it isn't readable or filtered yet.
