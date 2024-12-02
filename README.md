# vidi-natural-disaster-classification

- dataset
  - https://vididataset.github.io/VIDI/
  - pip install --upgrade youtube-dlp
  - yt-dlp --cookies-from-browser chrome --cookies cookies.txt

## Conda stuff

### Create a conda environment

conda create --name <environment-name> python=<version:2.7/3.5>

### To create a requirements.txt file:

conda list #Gives you list of packages used for the environment

conda list -e > requirements.txt #Save all the info about packages to your folder

### To export environment file

activate <environment-name>
conda env export > <environment-name>.yml

### For other person to use the environment

conda env create -f <environment-name>.yml
