#!/bin/bash

python manage.py makemigrations
python manage.py migrate

if [ "$DJANGO_SUPERUSER_USERNAME" ]
then
    python manage.py createsuperuser \
        --noinput \
        --username $DJANGO_SUPERUSER_USERNAME \
        --email $DJANGO_SUPERUSER_EMAIL
fi

if [ "$ENVIRONMENT" = "development" ]
then
    echo "Running in development mode"
    exec poetry run python manage.py runserver 0.0.0.0:8000
else
    echo "ENVIRONMENT variable is not set"
fi


$@