# Base Image: Choose a suitable one for your needs
FROM jupyter/datascience-notebook:latest
# Working directory within the container - jovyan is a default directory for jupyter notebooks named after a maintainer of the technology
WORKDIR /home/jovyan/work 
# Copy all required files from your project directory
COPY . /home/jovyan/work
# RUN pip install tensorflow pytorch
# Expose the Jupyter Notebook server to the host
EXPOSE 8888
CMD ["start-notebook.sh", "--NotebookApp.token=''"]

# docker build -t streamlit . 
# docker run -p 8501:8501 streamlit