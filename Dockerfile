# FROM tiangolo/uwsgi-nginx-flask:python3.8
FROM tethysts/tethys-dash-base:1.3

# RUN apt-get update && apt-get install -y libspatialindex-dev python-rtree

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy hello-cron file to the cron.d directory
# COPY tethys-cron /etc/cron.d/tethys-cron

# Give execution rights on the cron job
# RUN chmod 0644 /etc/cron.d/tethys-cron

# Apply cron job
# RUN crontab /etc/cron.d/tethys-cron

COPY ./app /app

CMD ["gunicorn", "--conf", "app/gunicorn_conf.py", "--bind", "0.0.0.0:80", "app.main:server"]
