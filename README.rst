Tethys Flow Naturalisation web app for Environment Southland
=============================================================

This git repository contains Dash app and docker files for running and deploying the flow naturalisation web app.
The app is built using Plotly and Dash. Then it is packaged up for deployment in Docker using the tiangolo/uwsgi-nginx-flask:python3.8 base image.
It is then deployed onto a Docker Swarm cluster using the docker-swarm.yml.
